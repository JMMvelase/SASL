#!/usr/bin/env python3
"""
train_unified.py
Train a unified temporal model (LSTM) on sequence .npy files listed in data/manifest.csv.

Usage:
    python src/train_unified.py --manifest data/manifest.csv --out models/unified_model.pth --epochs 30
"""
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# import your transforms (assumes src/transforms.py)
from transforms import ensure_63_from_xy, reorder_to_training_order, normalize_by_wrist, canonicalize_left_hand

# ---------------------------
# Config / Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest.csv")
parser.add_argument("--out", type=str, default="models/unified_model.pth", help="Model output path")
parser.add_argument("--metadata", type=str, default="models/unified_model.metadata.json", help="Metadata JSON")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seq-len", type=int, default=30, help="Target sequence length (padding/truncation)")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

MANIFEST = args.manifest
OUT_PATH = Path(args.out)
META_PATH = Path(args.metadata)
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
SEQ_LEN = args.seq_len
DEVICE = torch.device(args.device)
INPUT_DIM = 63  # 21*(x,y,z)

# ---------------------------
# Dataset + Collate
# ---------------------------
class SequenceDataset(Dataset):
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
        # ensure relative paths are valid
        self.df['file'] = self.df['file'].astype(str)
        # build label map
        labels = sorted(self.df['label'].unique())
        self.label2idx = {lbl: i for i, lbl in enumerate(labels)}
        self.idx2label = {i: lbl for lbl, i in self.label2idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = np.load(row['file']).astype(np.float32)  # shape (T, D)
        label = self.label2idx[row['label']]
        hand = row['hand'] if 'hand' in row.index else 'Unknown'
        return seq, label, hand

def collate_pad_mask(batch):
    seqs, labels, hands = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    D = seqs[0].shape[1]
    B = len(seqs)
    padded = np.zeros((B, maxlen, D), dtype=np.float32)
    mask = np.zeros((B, maxlen), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        padded[i, :L, :] = s
        mask[i, :L] = 1.0
    # optionally truncate/pad to SEQ_LEN (here we keep full maxlen)
    return torch.from_numpy(padded), torch.from_numpy(mask), torch.LongTensor(labels), torch.LongTensor(lengths)

# ---------------------------
# Model
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=INPUT_DIM, hidden=128, num_layers=2, num_classes=26, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, D)
        # we will use the final hidden state
        # (pack/pad optional, keep simple: we assume padded with zeros and final hidden state is fine)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # use last layer hidden state
        return out

# ---------------------------
# Load data
# ---------------------------
print("Loading manifest:", MANIFEST)
df = pd.read_csv(MANIFEST)
# split using manifest if not already split: use stratify on label
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_manifest = "data/train_manifest.tmp.csv"
val_manifest = "data/val_manifest.tmp.csv"
train_df.to_csv(train_manifest, index=False)
val_df.to_csv(val_manifest, index=False)

train_ds = SequenceDataset(train_manifest)
val_ds = SequenceDataset(val_manifest)

# keep label map from dataset
label_map = train_ds.label2idx
print("Label map:", label_map)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad_mask)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad_mask)

# ---------------------------
# Train loop
# ---------------------------
num_classes = len(train_ds.label2idx)
model = LSTMClassifier(input_size=INPUT_DIM, hidden=128, num_layers=2, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Training on", DEVICE)
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for padded, mask, labels, lengths in pbar:
        padded = padded.to(DEVICE)  # (B, T, D)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(padded)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=total_loss/total, acc=correct/total)

    train_acc = correct / total if total > 0 else 0.0

    # validation
    model.eval()
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        for padded, mask, labels, lengths in val_loader:
            padded = padded.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(padded)
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total if val_total > 0 else 0.0
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # save best
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "label_map": train_ds.idx2label if hasattr(train_ds, "idx2label") else train_ds.idx2label
        }, str(OUT_PATH))
        # metadata
        meta = {
            "input_dim": INPUT_DIM,
            "seq_len": SEQ_LEN,
            "label_map": train_ds.idx2label,
            "timestamp": time.time()
        }
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print("Saved best model:", OUT_PATH)

print("Training complete. Best val acc:", best_val_acc)
# cleanup temp manifests
os.remove(train_manifest)
os.remove(val_manifest)

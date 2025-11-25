# src/datasets.py
import numpy as np, torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, manifest_csv, label_map):
        import pandas as pd
        self.df = pd.read_csv(manifest_csv)
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = np.load(row['file']).astype(np.float32)  # shape (T, D)
        label = self.label_map[row['label']]
        hand = row['hand']
        return seq, label, hand

def collate_pad_mask(batch):
    # batch: list of (seq(T_i,D), label, hand)
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
    return torch.from_numpy(padded), torch.from_numpy(mask), torch.LongTensor(labels), torch.LongTensor(lengths), hands

# src/prepare_sequences.py
import os, csv, numpy as np, pandas as pd
from transforms import ensure_63_from_xy, reorder_to_training_order, normalize_by_wrist, canonicalize_left_hand

STATIC_CSV = "../data/sasl_landmarks.csv"
DYN_DIR = "../data/dynamic_dataset"
OUT_DIR = "data/sequences"
MANIFEST = "data/manifest.csv"
SEQ_LEN = 30

os.makedirs(OUT_DIR, exist_ok=True)

manifest_rows = []

# 1) convert dynamic files (already sequences of rows)
for fname in os.listdir(DYN_DIR):
    if not fname.endswith(".csv"): continue
    df = pd.read_csv(os.path.join(DYN_DIR, fname))
    # expect shape (SEQ_LEN, cols >= 3 + features)
    frames = []
    for _, row in df.iterrows():
        # row[3:] should be x0..x62 or similar; try numeric conversion
        flat = row.iloc[3:].astype(float).values
        flat63 = reorder_to_training_order(flat) if flat.size==63 else ensure_63_from_xy(flat)
        flat63 = normalize_by_wrist(flat63)
        # optional canonicalize
        hand = row['hand'] if 'hand' in row.index else 'Unknown'
        if hand == 'Left':
            flat63 = canonicalize_left_hand(flat63)
        frames.append(flat63)
    seq = np.stack(frames).astype(np.float32)
    out_name = f"seq_{fname.replace('.csv','')}.npy"
    out_path = os.path.join(OUT_DIR, out_name)
    np.save(out_path, seq)
    manifest_rows.append([os.path.relpath(out_path), row['letter'], hand, seq.shape[0], 30, "dynamic_recorder", "dynamic"])

# 2) convert static CSV rows into repeated sequences
df_static = pd.read_csv(STATIC_CSV)
for idx, row in df_static.iterrows():
    label = row['label']
    hand = row.get('hand', 'Unknown')
    # row values: assume x0..x20,y0..y20
    flat_xy = row.iloc[:-2].astype(float).values
    flat63 = ensure_63_from_xy(flat_xy)
    flat63 = normalize_by_wrist(flat63)
    if hand == 'Left':
        flat63 = canonicalize_left_hand(flat63)
    # repeat to SEQ_LEN
    seq = np.tile(flat63, (SEQ_LEN,1)).astype(np.float32)
    out_name = f"seq_static_{label}_{hand}_{idx}.npy"
    out_path = os.path.join(OUT_DIR, out_name)
    np.save(out_path, seq)
    manifest_rows.append([os.path.relpath(out_path), label, hand, seq.shape[0], 30, "static_from_single", "static"])

# Save manifest
with open(MANIFEST, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file","label","hand","seq_len","fps","source","tags"])
    writer.writerows(manifest_rows)

print("Done. sequences saved:", len(manifest_rows))

#!/usr/bin/env python3
"""
Show dataset statistics from manifest.csv
"""
import pandas as pd
import sys

MANIFEST = "data/manifest.csv"

if __name__ == "__main__":
    try:
        df = pd.read_csv(MANIFEST)
    except FileNotFoundError:
        print(f"❌ {MANIFEST} not found")
        sys.exit(1)
    
    if df.empty:
        print("❌ Manifest is empty")
        sys.exit(1)
    
    print(f"\n📊 Dataset Statistics ({len(df)} total sequences)\n")
    
    # Count by label
    print("=" * 50)
    print("Samples per Label:")
    print("=" * 50)
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    print(f"\nTotal unique labels: {len(label_counts)}")
    
    # Count by hand
    if 'hand' in df.columns:
        print("\n" + "=" * 50)
        print("Samples per Hand:")
        print("=" * 50)
        hand_counts = df['hand'].value_counts()
        for hand, count in hand_counts.items():
            print(f"  {hand}: {count}")
    
    # Count by label and hand
    if 'hand' in df.columns:
        print("\n" + "=" * 50)
        print("Samples per Label per Hand:")
        print("=" * 50)
        label_hand_counts = df.groupby(['label', 'hand']).size().unstack(fill_value=0)
        print(label_hand_counts)
    
    # Source breakdown
    if 'source' in df.columns:
        print("\n" + "=" * 50)
        print("Samples per Source:")
        print("=" * 50)
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
    
    print("\n")

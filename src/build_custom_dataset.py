#!/usr/bin/env python3
"""
build_custom_dataset.py

Builds a custom dataset by filtering the manifest.csv to include only desired labels.
This allows you to train on specific words or alphabets instead of the entire dataset.

Usage:
    # Train on specific words/letters:
    python src/build_custom_dataset.py --labels A B C HELLO WORLD --output data/custom_manifest.csv
    
    # Train on specific words:
    python src/build_custom_dataset.py --labels HELLO WORLD CAT DOG --output data/words_manifest.csv
    
    # Train on all alphabets (A-Z):
    python src/build_custom_dataset.py --labels A B C D E F G H I J K L M N O P Q R S T U V W X Y Z --output data/alphabet_manifest.csv

Then use this manifest with training:
    python src/train_unified.py --manifest data/custom_manifest.csv --out models/custom_model.pth
"""
import argparse
import pandas as pd
from pathlib import Path


def build_custom_dataset(manifest_path, labels, output_path):
    """
    Filter manifest to include only specified labels.
    
    Args:
        manifest_path: Path to the original manifest.csv
        labels: List of desired labels (case-insensitive)
        output_path: Path to save the filtered manifest.csv
    """
    # Read the original manifest
    df = pd.read_csv(manifest_path)
    
    # Normalize labels to uppercase for comparison
    labels_upper = [lbl.upper() for lbl in labels]
    
    print(f"Original manifest: {len(df)} samples")
    print(f"Available labels in manifest: {sorted(df['label'].unique())}")
    print(f"Filtering for labels: {labels_upper}")
    
    # Filter to only include desired labels
    df_filtered = df[df['label'].str.upper().isin(labels_upper)].copy()
    
    print(f"\nFiltered manifest: {len(df_filtered)} samples")
    
    # Show breakdown by label
    print("\nSamples per label:")
    for label in sorted(df_filtered['label'].unique()):
        count = len(df_filtered[df_filtered['label'] == label])
        print(f"  {label}: {count}")
    
    # Show breakdown by hand
    print("\nSamples per hand:")
    for hand in sorted(df_filtered['hand'].unique()):
        count = len(df_filtered[df_filtered['hand'] == hand])
        print(f"  {hand}: {count}")
    
    # Save the filtered manifest
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    print(f"\nSaved filtered manifest to: {output_path}")
    
    return df_filtered


def list_available_labels(manifest_path):
    """List all unique labels in the manifest."""
    df = pd.read_csv(manifest_path)
    labels = sorted(df['label'].unique())
    print("Available labels in dataset:")
    for label in labels:
        count = len(df[df['label'] == label])
        print(f"  {label}: {count} samples")
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Build a custom dataset manifest with only desired labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on alphabets A, B, C
  python src/build_custom_dataset.py --labels A B C --output data/abc_manifest.csv
  
  # Train on words
  python src/build_custom_dataset.py --labels HELLO WORLD --output data/words_manifest.csv
  
  # List all available labels
  python src/build_custom_dataset.py --list
        """
    )
    
    parser.add_argument("--manifest", type=str, default="data/manifest.csv",
                        help="Path to original manifest.csv")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="List of desired labels (space-separated, case-insensitive)")
    parser.add_argument("--output", type=str, default="data/custom_manifest.csv",
                        help="Path to save filtered manifest")
    parser.add_argument("--list", action="store_true",
                        help="List all available labels in the manifest")
    
    args = parser.parse_args()
    
    # List available labels if requested
    if args.list:
        list_available_labels(args.manifest)
        return
    
    # Build custom dataset
    if not args.labels:
        print("Error: --labels required (or use --list to see available labels)")
        parser.print_help()
        return
    
    build_custom_dataset(args.manifest, args.labels, args.output)


if __name__ == "__main__":
    main()

# Custom Dataset & Training Guide

## Overview
Your SASL (Sign Language) recognition system now supports training on custom subsets of your dataset. You can select specific alphabets, words, or any combination you want to train on.

## Quick Start

### 1. List Available Labels in Your Dataset
```powershell
cd C:\Users\me\SASL\SASL
python src/build_custom_dataset.py --list
```

This will show all available labels and their sample counts.

### 2. Create a Custom Dataset for Specific Labels

#### Example 1: Train on Alphabets A, B, C
```powershell
python src/build_custom_dataset.py --labels A B C --output data/abc_manifest.csv
```

#### Example 2: Train on Words (if you have them)
```powershell
python src/build_custom_dataset.py --labels HELLO WORLD --output data/words_manifest.csv
```

#### Example 3: Train on All Alphabets A-Z
```powershell
python src/build_custom_dataset.py --labels A B C D E F G H I J K L M N O P Q R S T U V W X Y Z --output data/alphabet_manifest.csv
```

#### Example 4: Mix of Alphabets and Words
```powershell
python src/build_custom_dataset.py --labels A B C HELLO WORLD CAT --output data/mixed_manifest.csv
```

### 3. Train Your Model on the Custom Dataset

After creating a custom manifest, train using:
```powershell
python src/train_unified.py --manifest data/custom_manifest.csv --out models/custom_model.pth --metadata models/custom_model.metadata.json --epochs 30
```

### 4. Use Your Custom Trained Model for Inference

```powershell
python src/inference_unified.py --model models/custom_model.pth --metadata models/custom_model.metadata.json
```

---

## Detailed Workflow Examples

### Scenario 1: Build a Simple 3-Letter Recognition System

```powershell
# Step 1: Create manifest with just A, B, C
python src/build_custom_dataset.py --labels A B C --output data/abc_manifest.csv

# Step 2: Train model on ABC dataset
python src/train_unified.py --manifest data/abc_manifest.csv --out models/abc_model.pth --metadata models/abc_model.metadata.json --epochs 50

# Step 3: Run inference with the trained model
python src/inference_unified.py --model models/abc_model.pth --metadata models/abc_model.metadata.json
```

### Scenario 2: Build a Word Recognition System

```powershell
# Step 1: Create manifest with words
python src/build_custom_dataset.py --labels HELLO WORLD YES NO --output data/words_manifest.csv

# Step 2: Check the dataset
python src/build_custom_dataset.py --manifest data/words_manifest.csv --list

# Step 3: Train model
python src/train_unified.py --manifest data/words_manifest.csv --out models/words_model.pth --metadata models/words_model.metadata.json --epochs 50

# Step 4: Run inference
python src/inference_unified.py --model models/words_model.pth --metadata models/words_model.metadata.json
```

### Scenario 3: Progressive Training (Start with few, expand later)

```powershell
# Stage 1: Train on A-F
python src/build_custom_dataset.py --labels A B C D E F --output data/stage1_manifest.csv
python src/train_unified.py --manifest data/stage1_manifest.csv --out models/stage1_model.pth --epochs 30

# Stage 2: Expand to A-L
python src/build_custom_dataset.py --labels A B C D E F G H I J K L --output data/stage2_manifest.csv
python src/train_unified.py --manifest data/stage2_manifest.csv --out models/stage2_model.pth --epochs 30

# Stage 3: Full alphabet
python src/build_custom_dataset.py --labels A B C D E F G H I J K L M N O P Q R S T U V W X Y Z --output data/full_manifest.csv
python src/train_unified.py --manifest data/full_manifest.csv --out models/full_model.pth --epochs 30
```

---

## Training Command Reference

### Basic Training
```powershell
python src/train_unified.py --manifest data/custom_manifest.csv --out models/custom_model.pth
```

### Training with More Epochs (Better Accuracy)
```powershell
python src/train_unified.py --manifest data/custom_manifest.csv --out models/custom_model.pth --epochs 50
```

### Training with Larger Batch Size (Faster, needs more GPU memory)
```powershell
python src/train_unified.py --manifest data/custom_manifest.csv --out models/custom_model.pth --batch-size 32 --epochs 50
```

### Training on GPU (if available)
```powershell
python src/train_unified.py --manifest data/custom_manifest.csv --out models/custom_model.pth --device cuda --epochs 50
```

### Training with Different Sequence Length
```powershell
python src/train_unified.py --manifest data/custom_manifest.csv --out models/custom_model.pth --seq-len 30 --epochs 50
```

---

## Inference Command Reference

### Basic Inference
```powershell
python src/inference_unified.py --model models/custom_model.pth --metadata models/custom_model.metadata.json
```

### Inference with Confidence Threshold
```powershell
python src/inference_unified.py --model models/custom_model.pth --metadata models/custom_model.metadata.json --confidence-threshold 0.5
```

### Inference with Smoothing
```powershell
python src/inference_unified.py --model models/custom_model.pth --metadata models/custom_model.metadata.json --smooth-window 5
```

---

## Understanding the Output

When you run `build_custom_dataset.py`, it will show:

```
Original manifest: 1260 samples
Available labels in manifest: ['A', 'B', 'C', ..., 'Z']
Filtering for labels: ['A', 'B', 'C']

Filtered manifest: 150 samples

Samples per label:
  A: 50
  B: 50
  C: 50

Samples per hand:
  Left: 75
  Right: 75

Saved filtered manifest to: data/abc_manifest.csv
```

This tells you:
- Total samples before filtering
- Which labels exist in your dataset
- How many samples you'll use for training
- Distribution across labels and hands

---

## Tips & Best Practices

1. **Start Small**: Begin training with 3-5 labels to verify the pipeline works
2. **Balance**: Try to have similar sample counts across different labels
3. **Both Hands**: If training on both hands, the model should be more robust
4. **Epochs**: Start with 30 epochs, increase to 50-100 for better accuracy
5. **Monitor**: Watch the training output for accuracy improving
6. **Test**: Always test with inference after training

---

## Troubleshooting

### Error: "Available labels in manifest: []"
- Your manifest.csv might be empty or corrupted
- Try: `python src/build_custom_dataset.py --list` to see what's available

### Error: "No samples matched your labels"
- The labels you specified don't exist in your dataset
- Use `--list` flag to see available labels (they're case-sensitive by default)

### Training is very slow
- Reduce batch size: `--batch-size 8`
- Use GPU if available: `--device cuda`
- Use fewer labels to start

### Low accuracy during training
- Add more epochs: `--epochs 100`
- Make sure you have enough samples per label (at least 5-10)
- Check that your video recordings are of good quality

---

## File Locations

- Original manifest: `data/manifest.csv`
- Custom manifests: `data/custom_manifest.csv`, `data/abc_manifest.csv`, etc.
- Models: `models/custom_model.pth`, `models/abc_model.pth`, etc.
- Metadata: `models/custom_model.metadata.json`, etc.

import argparse
import cv2
import mediapipe as mp
import csv
import os
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import shutil
import time

# === Setup ===
LETTERS = [chr(i) for i in range(65, 91)]  # A-Z
  # reference images (A.jpg, B.jpg, ...)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "sasl_landmarks.csv")
DYNAMIC_DATA_DIR = os.path.join(DATA_DIR, "dynamic_dataset")
os.makedirs(DYNAMIC_DATA_DIR, exist_ok=True)
MODEL_PATH = "sasl_model.pkl"

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Create CSV if it doesn't exist
# Format: x0,x1,...,x20,y0,y1,...,y20,z0,z1,...,z20,label,hand (63 features + 2 labels)
if not os.path.isfile(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ([f"x{i}" for i in range(21)] + 
                  [f"y{i}" for i in range(21)] + 
                  [f"z{i}" for i in range(21)] + 
                  ["label", "hand"])
        writer.writerow(header)


def show_stats(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        print("No 'label' column in CSV.")
        return
    counts = df['label'].value_counts()
    print("Dataset label counts:")
    print(counts)


def import_csv(src_csv, dest_csv=CSV_PATH, mode='merge', overwrite_label=None, force=False):
    """Import rows from src_csv into dest_csv.
    mode: 'merge' (append) or 'overwrite' (remove rows for overwrite_label then append)
    If overwrite_label is provided, rows in dest_csv with that label will be removed first.
    """
    if not os.path.exists(src_csv):
        raise FileNotFoundError(src_csv)

    df_src = pd.read_csv(src_csv)
    if df_src.shape[0] == 0:
        print("Source CSV empty, nothing to import.")
        return

    # ensure header consistency: if necessary, drop extra columns and keep required ones
    required_cols = ([f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label", "hand"])
    # keep only columns that exist in src and are required (allow extra columns in src)
    keep_cols = [c for c in required_cols if c in df_src.columns]
    df_src = df_src[keep_cols]

    if os.path.exists(dest_csv):
        df_dest = pd.read_csv(dest_csv)
    else:
        # create destination with header
        df_dest = pd.DataFrame(columns=required_cols)

    if overwrite_label:
        if not force:
            raise RuntimeError("Refusing to overwrite existing data without --force-overwrite. Use force=True to allow destructive overwrite.")
        # filter only rows in source that match the overwrite label
        rows_to_add = df_src[df_src['label'] == overwrite_label]
        # backup destination before destructive change
        if os.path.exists(dest_csv):
            bak_name = f"{dest_csv}.bak.{int(time.time())}"
            shutil.copy(dest_csv, bak_name)
            print(f"Backup of destination CSV created: {bak_name}")
        # remove existing rows for that label in dest
        df_dest = df_dest[df_dest['label'] != overwrite_label]
        if rows_to_add.shape[0] == 0:
            print(f"Warning: no rows with label='{overwrite_label}' found in source CSV.")
    else:
        rows_to_add = df_src

    # Append rows (rows_to_add may have missing columns; align columns)
    rows_to_add = rows_to_add.reindex(columns=df_dest.columns)
    df_new = pd.concat([df_dest, rows_to_add], ignore_index=True)
    df_new.to_csv(dest_csv, index=False)
    print(f"Imported {rows_to_add.shape[0]} rows into {dest_csv} (mode={mode}, overwrite_label={overwrite_label})")

# === Helper function ===
def extract_landmarks(hand_landmarks, shape):
    """Extract normalized x,y,z coordinates (normalized by wrist->middle_mcp distance)"""
    # Get raw normalized coordinates (0..1)
    pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    pts = np.array(pts)
    
    # Normalize by wrist->middle_mcp distance
    wx, wy, wz = pts[0]
    mx, my, mz = pts[9]
    scale = math.hypot(mx - wx, my - wy)
    
    if scale < 1e-6:
        # Fallback: use max distance from wrist
        dists = np.linalg.norm(pts - np.array([wx, wy, wz]), axis=1)
        scale = max(1e-3, dists.max())
    
    # Normalize all points
    pts[:, 0] = (pts[:, 0] - wx) / scale
    pts[:, 1] = (pts[:, 1] - wy) / scale
    pts[:, 2] = (pts[:, 2] - wz) / scale
    
    # Flatten to [x0,x1,...,x20,y0,y1,...,y20,z0,z1,...,z20]
    x_list = pts[:, 0].tolist()
    y_list = pts[:, 1].tolist()
    z_list = pts[:, 2].tolist()
    return x_list, y_list, z_list


def write_dynamic_sequence(rows, label, hand, output_dir=DYNAMIC_DATA_DIR):
    if not rows:
        print("No frames recorded, nothing saved.")
        return
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{label}_{hand}_{int(time.time())}.csv"
    out_path = os.path.join(output_dir, filename)
    header = ["letter", "hand", "frame"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"✅ Saved dynamic sequence: {out_path}")

# === Import numpy ===
import numpy as np

# === Camera setup ===
import sys

# --- CLI: allow showing dataset, importing CSVs, overwriting label rows ---
parser = argparse.ArgumentParser(description="SASL data capture and CSV utilities")
parser.add_argument("--show", action="store_true", help="Show dataset label counts and exit")
parser.add_argument("--import-csv", type=str, help="Path to CSV to import into dataset CSV")
parser.add_argument("--overwrite-label", type=str, default=None, help="If set, replace existing rows for this label with rows from source CSV")
parser.add_argument("--force-overwrite", action="store_true", help="Allow destructive overwrite when using --overwrite-label (creates a backup first)")
parser.add_argument("--dynamic", action="store_true", help="Capture dynamic sign sequences instead of single-frame static samples")
parser.add_argument("--no-capture", action="store_true", help="Don't open webcam capture (useful when running import/show modes)")
args = parser.parse_args()

if args.show:
    show_stats(CSV_PATH)
    sys.exit(0)

if args.import_csv:
    src = args.import_csv
    # mode is determined by whether overwrite_label is set
    mode = 'overwrite' if args.overwrite_label else 'merge'
    try:
        import_csv(src, dest_csv=CSV_PATH, mode=mode, overwrite_label=args.overwrite_label, force=args.force_overwrite)
    except RuntimeError as e:
        print(str(e))
        print("No data was modified. To allow overwrite, re-run with --force-overwrite.")
        sys.exit(1)
    # after importing, if user doesn't want capture, exit
    if args.no_capture:
        print("Import complete. Exiting (no capture).")
        sys.exit(0)

# proceed to camera only if not suppressed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

letter_index = 0
recording = False
sequence_rows = []
frame_index = 0
if args.dynamic:
    print("📷 SASL Trainer: R = start recording, S = stop/save, N = next letter, ESC = quit")
else:
    print("📷 SASL Trainer: SPACE = capture static frame, N = next letter, ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    letter = LETTERS[letter_index]

    # Draw landmarks
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # On-screen instructions
    cv2.putText(frame, f"Letter: {letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if args.dynamic:
        mode_text = "R=Record | S=Stop | N=Next | ESC=Quit"
        if recording:
            cv2.putText(frame, f"REC {len(sequence_rows)} frames", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        mode_text = "SPACE=Capture | N=Next Letter | ESC=Quit"
    cv2.putText(frame, mode_text,
                (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("SASL Trainer", frame)
    key = cv2.waitKey(1) & 0xFF

    if args.dynamic:
        if key == ord('r') and not recording:
            recording = True
            sequence_rows = []
            frame_index = 0
            print(f"🔴 Recording dynamic sequence for {letter}")

        if recording and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_list, y_list, z_list = extract_landmarks(hand_landmarks, frame.shape)
            if results.multi_handedness and len(results.multi_handedness) > 0:
                handedness_label = results.multi_handedness[0].classification[0].label
            else:
                handedness_label = "Unknown"
            row = [letter, handedness_label, frame_index] + x_list + y_list + z_list
            sequence_rows.append(row)
            frame_index += 1

        if key == ord('s') and recording:
            recording = False
            if sequence_rows:
                write_dynamic_sequence(sequence_rows, letter, sequence_rows[-1][1])
            else:
                print("No frames were recorded.")

    else:
        if key == 32 and results.multi_hand_landmarks:  # SPACE
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                x_list, y_list, z_list = extract_landmarks(hand_landmarks, frame.shape)
                if results.multi_handedness and i < len(results.multi_handedness):
                    handedness_label = results.multi_handedness[i].classification[0].label
                else:
                    handedness_label = "Unknown"
                row = x_list + y_list + z_list + [letter, handedness_label]
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"✅ Saved {letter} ({handedness_label})")

    if key == ord("n"):
        letter_index = (letter_index + 1) % len(LETTERS)
        print(f"➡️ Now practicing {LETTERS[letter_index]}")

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

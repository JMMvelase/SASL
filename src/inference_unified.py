#!/usr/bin/env python3
"""
inference_unified.py
Realtime inference using the unified temporal model.

Usage:
    python src/inference_unified.py --model models/unified_model.pth --metadata models/unified_model.metadata.json
"""
import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import json
import math

from transforms import reorder_to_training_order, ensure_63_from_xy, normalize_by_wrist, canonicalize_left_hand

# ---------------------------
# Config / CLI
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="models/unified_model.pth")
parser.add_argument("--metadata", type=str, default="models/unified_model.metadata.json")
parser.add_argument("--seq-len", type=int, default=30)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--smooth-window", type=int, default=3, help="moving average smoothing window for predictions")
parser.add_argument("--confidence-threshold", type=float, default=0.3)
args = parser.parse_args()

MODEL_PATH = args.model
META_PATH = args.metadata
SEQ_LEN = args.seq_len
DEVICE = torch.device(args.device)
SMOOTH_WINDOW = args.smooth_window
CONF_THRESH = args.confidence_threshold

# ---------------------------
# Model class (must match training)
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=63, hidden=128, num_layers=2, num_classes=26):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# ---------------------------
# Load model + metadata
# ---------------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state = ckpt.get("model_state_dict", ckpt)
# load metadata file for label map
with open(META_PATH, "r") as f:
    meta = json.load(f)
label_map = meta.get("label_map")
if isinstance(label_map, dict):
    # label_map might be idx->label; ensure idx->label
    idx2label = {int(k): v for k, v in label_map.items()}
else:
    # fallback to alphabet order
    idx2label = {i: c for i, c in enumerate(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))}

num_classes = len(idx2label)
model = LSTMClassifier(input_size=63, hidden=128, num_layers=2, num_classes=num_classes).to(DEVICE)
model.load_state_dict(state)
model.eval()
print("Loaded unified model:", MODEL_PATH)

# ---------------------------
# Mediapipe setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ---------------------------
# Helpers
# ---------------------------
def flat_from_mediapipe(lm_list):
    flat = np.array([coord for p in lm_list for coord in (p.x, p.y, p.z)], dtype=np.float32)
    if flat.size == 63:
        return reorder_to_training_order(flat)  # ensure training order (x..,y..,z..)
    elif flat.size == 42:
        return ensure_63_from_xy(flat)
    else:
        raise ValueError("unexpected landmark size")

def preprocess_frame_from_mediapipe(lm_list, hand_label):
    flat = flat_from_mediapipe(lm_list)
    flat = normalize_by_wrist(flat)
    if hand_label.lower().startswith("l"):
        flat = canonicalize_left_hand(flat)
    return flat

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---------------------------
# Rolling buffer, smoothing
# ---------------------------
buffer = deque(maxlen=SEQ_LEN)
pred_history = deque(maxlen=SMOOTH_WINDOW)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("Starting inference. Press ESC to quit.")
last_announced = None
last_time = 0.0
DEBOUNCE_SEC = 0.6

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    pred_text = "no_hand"
    handedness = "Unknown"

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        if res.multi_handedness:
            handedness = res.multi_handedness[0].classification[0].label

        try:
            feat = preprocess_frame_from_mediapipe(lm.landmark, handedness)
        except Exception as e:
            print("Preprocess error:", e)
            continue

        buffer.append(feat)

        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        if len(buffer) >= SEQ_LEN:
            seq = np.stack(list(buffer)[-SEQ_LEN:]).astype(np.float32)  # (T, D)
            with torch.no_grad():
                xin = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)  # (1,T,D)
                out = model(xin)
                probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                pred_label = idx2label.get(pred_idx, "?")

            # smoothing
            pred_history.append((pred_label, conf))
            # compute moving average by labels: choose mode with average conf
            # simple heur: if last SMOOTH_WINDOW preds agree, announce
            labels = [p for p, c in pred_history]
            if len(labels) >= SMOOTH_WINDOW and all(x == labels[-1] for x in labels[-SMOOTH_WINDOW:]):
                if conf >= CONF_THRESH:
                    pred_text = f"{pred_label} ({conf:.2f})"
                    now = time.time()
                    if pred_text != last_announced and (now - last_time) > DEBOUNCE_SEC:
                        print(f"[{time.strftime('%H:%M:%S')}] {pred_text}")
                        last_announced = pred_text
                        last_time = now
            else:
                pred_text = f"{pred_label} ({conf:.2f})"
        else:
            pred_text = f"collecting {len(buffer)}/{SEQ_LEN}"

    # overlays
    cv2.putText(frame, pred_text, (10, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(frame, f"Hand: {handedness}", (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Unified Inference", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

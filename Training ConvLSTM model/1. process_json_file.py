import os
import json
import numpy as np
import pandas as pd
import pickle
from glob import glob

# === Cài đặt ===
JSON_DIR = "JSON file/"
OUTPUT_DIR = "data/"
SEQUENCE_LENGTH = 30
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
NUM_KEYPOINTS = 17

# -- Ánh xạ tên hành động
ACTION_NAME_MAP = {
    (0, 4): "Walking",
    (0, 21): "Transition state",
    (0, 28): "Lay down",
    (0, 31): "Standing",
    (0, 33): "Check-in",
    (0, 34): "Check-out",
    (0, 35): "Carry object",
    (0, 36): "Work in progress",
    (0, 37): "Falling down",
    (22, 31): "Pick up a tool",
    (23, 31): "Put down a tool",
    (24, 31): "setup machine"
}

LABEL_MAP = {k: f"action_{str(i+1).zfill(2)}" for i, k in enumerate(ACTION_NAME_MAP.keys())}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_keypoints(keypoints, bbox):
    x_min, y_min, w, h = bbox
    normed = []
    for point in keypoints:
        x, y = point[0], point[1]
        x_n = (x - x_min) / w if w > 1 else x / IMAGE_WIDTH
        y_n = (y - y_min) / h if h > 1 else y / IMAGE_HEIGHT
        normed.append([x_n, y_n])
    return normed  # 17 x 2

all_sequences = []
all_labels = []
action_counts = {}

json_files = sorted(glob(os.path.join(JSON_DIR, "P700C001A*.json")))

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    anns = data["annotations"]
    track_dict = {}

    for ann in anns:
        aid = (ann["action_id"]["action_upper"], ann["action_id"]["action_lower"])
        if aid == (0, 0) or aid not in ACTION_NAME_MAP:
            continue  # Bỏ nhãn null hoặc không dùng

        tid = ann.get("track_id", 0)
        if tid not in track_dict:
            track_dict[tid] = []

        entry = {
            "image_id": ann["image_id"],
            "keypoints": ann["keypoints"],
            "bbox": ann["bbox"],
            "label_key": aid
        }
        track_dict[tid].append(entry)

    # Lọc chuỗi 30 frame cùng nhãn
    for tid, frames in track_dict.items():
        frames = sorted(frames, key=lambda x: x["image_id"])
        for i in range(len(frames) - SEQUENCE_LENGTH + 1):
            subseq = frames[i:i+SEQUENCE_LENGTH]
            labels = [f["label_key"] for f in subseq]
            if len(set(labels)) == 1:
                kp_seq = [normalize_keypoints(f["keypoints"], f["bbox"]) for f in subseq]
                all_sequences.append(kp_seq)
                all_labels.append(labels[0])
                action_counts[labels[0]] = action_counts.get(labels[0], 0) + 1

    print(f"✅ Processed: {os.path.basename(json_file)}")

# === Lưu file ===
X = np.array(all_sequences)  # (N, 30, 17, 2)
Y = np.array([LABEL_MAP[lbl] for lbl in all_labels])

np.save(os.path.join(OUTPUT_DIR, "processed_features.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "processed_features_labels.npy"), Y)
with open(os.path.join(OUTPUT_DIR, "label_mapping.pkl"), "wb") as f:
    pickle.dump(LABEL_MAP, f)

# Đổi action_counts thành DataFrame
counts_df = pd.DataFrame([
    {"action_upper": k[0], "action_lower": k[1], "label": LABEL_MAP[k], "count": v}
    for k, v in action_counts.items()
])
counts_df.to_csv(os.path.join(OUTPUT_DIR, "action_counts.csv"), index=False)

print(f"\n✅ Done. Total sequences: {len(X)}, shape: {X.shape}")
print(f"✅ Labels: {set(Y)}")

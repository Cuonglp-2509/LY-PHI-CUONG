import os
import pandas as pd
import numpy as np
from glob import glob
import pickle

# -- Settings
KEYPOINTS_DIR = "output/keypoints/"
ATTRIBUTES_DIR = "output/excel/"
OUTPUT_X_PATH = "data/processed_features.npy"
OUTPUT_Y_PATH = "data/processed_features_labels.npy"
OUTPUT_LABEL_MAP_PATH = "data/label_mapping.pkl"
OUTPUT_ACTION_COUNTS_PATH = "data/action_counts.csv"
SEQUENCE_LENGTH = 20
STEP = 1
EXCLUDED_ACTION_PAIRS = [(0, 0)]  # Chỉ loại bỏ Null
NUM_KEYPOINTS = 17
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
AUGMENT_FACTOR = 15  # Nhân bản cho các hành vi hiếm
FALLING_DOWN_PAIR = (0, 37)  # Falling down
LAY_DOWN_PAIR = (0, 28)  # Lay down
PICK_UP_OBJECT_PAIR = (22, 31)  # Pick up object
PUT_DOWN_OBJECT_PAIR = (23, 31)  # Put down object
STANDING_PAIR = (0, 21)  # Standing

# -- Hàm chuẩn hóa keypoints
def normalize_keypoints(keypoints, bbox):
    """Chuẩn hóa tọa độ keypoints, giữ confidence."""
    x_min, y_min, width, height = bbox
    normalized = []
    for i in range(0, len(keypoints), 3):
        x, y, c = keypoints[i], keypoints[i+1], keypoints[i+2]
        x_norm = (x - x_min) / width if width > 1 else (x / IMAGE_WIDTH)
        y_norm = (y - y_min) / height if height > 1 else (y / IMAGE_HEIGHT)
        normalized.extend([x_norm, y_norm, c])
    return normalized

# -- Hàm thêm nhiễu cho hành vi hiếm
def augment_rare_action(keypoints, bbox, augment_factor=20):
    """Nhân bản và thêm nhiễu cho keypoints."""
    augmented_keypoints = []
    for _ in range(augment_factor):
        noise = np.random.normal(0, 0.05, size=len(keypoints))  # Nhiễu ±5%
        noise[2::3] = 0  # Không thêm nhiễu vào confidence
        aug_keypoints = keypoints + noise
        norm_keypoints = normalize_keypoints(aug_keypoints, bbox)
        augmented_keypoints.append(norm_keypoints)
    return augmented_keypoints

# -- Hàm trích xuất chuỗi
def extract_sequences(keypoints_df, attributes_df, sequence_length, step, excluded_pairs):
    """Trích xuất chuỗi keypoints và nhãn."""
    sequences = []
    labels = []
    
    keypoints_df = keypoints_df.sort_values("image_id")
    attributes_df = attributes_df.sort_values("image_id")
    
    if len(keypoints_df) != len(attributes_df):
        print(f"Warning: Mismatch in number of frames: keypoints={len(keypoints_df)}, attributes={len(attributes_df)}")
        return [], []
    
    for track_id in keypoints_df["track_id"].unique():
        track_kp = keypoints_df[keypoints_df["track_id"] == track_id]
        track_attr = attributes_df[attributes_df["track_id"] == track_id]
        
        # Debug: In số khung hình và hành vi
        action_pairs = track_attr[["action_upper", "action_lower"]].apply(tuple, axis=1).value_counts()
        print(f"Track ID {track_id}: {len(track_kp)} frames, actions: {action_pairs.to_dict()}")
        
        # Xử lý các hành vi hiếm (Falling down, Lay down, Pick up object, Put down object)
        rare_pairs = [FALLING_DOWN_PAIR, LAY_DOWN_PAIR, PICK_UP_OBJECT_PAIR, PUT_DOWN_OBJECT_PAIR,STANDING_PAIR]
        has_rare_action = False
        for pair in rare_pairs:
            if ((track_attr["action_upper"] == pair[0]) & (track_attr["action_lower"] == pair[1])).any():
                has_rare_action = True
                break
        
        if has_rare_action:
            for i in range(len(track_kp)):
                kp_row = track_kp.iloc[i]
                attr_row = track_attr.iloc[i]
                action_pair = (attr_row["action_upper"], attr_row["action_lower"])
                
                if action_pair in rare_pairs:
                    keypoints = kp_row[[f"{coord}{k+1}" for k in range(NUM_KEYPOINTS) for coord in ["x", "y", "c"]]].values
                    bbox = kp_row[["bbox_x_min", "bbox_y_min", "bbox_width", "bbox_height"]].values
                    aug_keypoints_list = augment_rare_action(keypoints, bbox, AUGMENT_FACTOR)
                    
                    # Tạo chuỗi bằng cách lặp lại các mẫu augmented
                    for _ in range(AUGMENT_FACTOR):
                        sequence = aug_keypoints_list * (sequence_length // len(aug_keypoints_list) + 1)
                        sequence = sequence[:sequence_length]
                        if len(sequence) == sequence_length:
                            sequences.append(sequence)
                            labels.append(action_pair)
                            print(f"Created sequence for {action_pair}")
        else:
            for i in range(0, len(track_kp) - sequence_length + 1, step):
                sequence = []
                action_pairs = []
                
                for j in range(i, i + sequence_length):
                    kp_row = track_kp.iloc[j]
                    attr_row = track_attr.iloc[j]
                    
                    keypoints = kp_row[[f"{coord}{k+1}" for k in range(NUM_KEYPOINTS) for coord in ["x", "y", "c"]]].values
                    bbox = kp_row[["bbox_x_min", "bbox_y_min", "bbox_width", "bbox_height"]].values
                    norm_keypoints = normalize_keypoints(keypoints, bbox)
                    sequence.append(norm_keypoints)
                    action_pairs.append((attr_row["action_upper"], attr_row["action_lower"]))
                
                if len(sequence) == sequence_length and action_pairs[-1] not in excluded_pairs:
                    sequences.append(sequence)
                    labels.append(action_pairs[-1])
    
    return sequences, labels

# -- Hàm đếm số frame
def count_action_frames(attributes_df):
    """Đếm số frame cho mỗi cặp action."""
    action_counts = attributes_df.groupby(["action_upper", "action_lower"]).size().reset_index(name="frame_count")
    return action_counts

# -- Hàm tạo nhãn action_XX
def create_action_labels(all_labels):
    """Tạo nhãn dạng action_XX."""
    unique_pairs = sorted(list(set(all_labels)))
    label_map = {pair: f"action_{str(idx + 1).zfill(2)}" for idx, pair in enumerate(unique_pairs)}
    return label_map

# -- Main
def main():
    os.makedirs(os.path.dirname(OUTPUT_X_PATH), exist_ok=True)
    all_sequences = []
    all_labels = []
    all_action_counts = []
    
    keypoint_files = glob(os.path.join(KEYPOINTS_DIR, "*.csv"))
    keypoint_files.sort()
    
    for kp_file in keypoint_files:
        json_filename = os.path.basename(kp_file).replace("_keypoints.csv", "")
        attr_file = os.path.join(ATTRIBUTES_DIR, f"{json_filename}_attributes.xlsx")
        
        if not os.path.exists(attr_file):
            print(f"Warning: Attributes file {attr_file} not found")
            continue
        
        print(f"Processing {kp_file} and {attr_file}")
        
        keypoints_df = pd.read_csv(kp_file)
        attributes_df = pd.read_excel(attr_file)
        
        sequences, labels = extract_sequences(keypoints_df, attributes_df, SEQUENCE_LENGTH, STEP, set(EXCLUDED_ACTION_PAIRS))
        all_sequences.extend(sequences)
        all_labels.extend(labels)
        
        action_counts = count_action_frames(attributes_df)
        action_counts["source_file"] = json_filename
        all_action_counts.append(action_counts)
    
    if not all_sequences:
        print("No valid sequences found")
        return
    
    # Tạo nhãn
    label_map = create_action_labels(all_labels)
    new_labels = [label_map[pair] for pair in all_labels]
    
    # Chuyển thành numpy array
    X = np.array(all_sequences)  # Shape: (num_sequences, seq_len, num_keypoints * 3)
    Y = np.array(new_labels)
    
    # Lưu dữ liệu
    np.save(OUTPUT_X_PATH, X)
    np.save(OUTPUT_Y_PATH, Y)
    
    with open(OUTPUT_LABEL_MAP_PATH, "wb") as f:
        pickle.dump(label_map, f)
    
    action_counts_df = pd.concat(all_action_counts, ignore_index=True)
    action_counts_df = action_counts_df.groupby(["action_upper", "action_lower"]).sum().reset_index()
    action_counts_df.to_csv(OUTPUT_ACTION_COUNTS_PATH, index=False)
    
    print(f"Saved {len(X)} sequences to {OUTPUT_X_PATH}")
    print(f"Saved {len(Y)} labels to {OUTPUT_Y_PATH}")
    print(f"Saved label mapping to {OUTPUT_LABEL_MAP_PATH}")
    print(f"Saved action counts to {OUTPUT_ACTION_COUNTS_PATH}")
    print(f"X shape: {X.shape}")
    print(f"Label mapping: {label_map}")

if __name__ == "__main__":
    main()
import json
import cv2
import numpy as np
import os
from glob import glob

# -- Settings
JSON_FILE = "JSON file/P700C001A0089R001.json"
IMAGE_DIR = os.path.join("images", "P700", "C001", "A0089", "R001")  # Cross-platform path
OUTPUT_VIDEO = "A0089R001.mp4"
FPS = 30  # Frames per second for the video

# -- Load JSON data
try:
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at {JSON_FILE}")
    exit(1)

# -- Create image_id to file_name mapping
image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# -- Get all image files and sort them
image_files = sorted(glob(os.path.join(IMAGE_DIR, "P700C001A0089R001F*.jpg")))
if not image_files:
    print(f"Error: No image files found in {IMAGE_DIR}")
    exit(1)

# -- Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
frame = cv2.imread(image_files[0])
if frame is None:
    print(f"Error: Could not load first image: {image_files[0]}")
    exit(1)
height, width, _ = frame.shape
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))

# -- Process each image and create video
total_frames = len(image_files)
for idx, img_file in enumerate(image_files, 1):
    # Load image
    image = cv2.imread(img_file)
    if image is None:
        print(f"Warning: Could not load image: {img_file}, skipping...")
        continue
    
    # Find corresponding image_id and annotation
    try:
        image_id = next(img["id"] for img in data["images"] if img["file_name"] == os.path.basename(img_file))
        annotation = next(ann for ann in data["annotations"] if ann["image_id"] == image_id)
    except StopIteration:
        print(f"Warning: No annotation found for image: {img_file}, skipping annotations...")
        out.write(image)  # Write unannotated frame
        continue
    
    
    # Write frame to video
    out.write(image)
    
    # Progress feedback
    print(f"Processing frame {idx}/{total_frames} ({(idx/total_frames)*100:.1f}%)")

# -- Release video writer
out.release()
print(f"Saved output video to {OUTPUT_VIDEO}")

# -- Optional: Display last frame (uncomment to show)
# cv2.imshow("Last Frame with Keypoints", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
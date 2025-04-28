import os
import cv2
import shutil
from tqdm import tqdm  # <-- make sure tqdm is installed: pip install tqdm

# Paths
input_dir = r"D:\archive (2)\dr_unified_v2\dr_unified_v2\val\0"  # folder with mixed images
original_dir = r"D:\archive (2)\dr_unified_v2\dr_unified_v1\train\0"
processed_dir = r"D:\archive (2)\dr_unified_v2\dr_gray\val\0"

# Create output directories if they don't exist
os.makedirs(original_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Thresholds (tune if needed)
min_size_for_original = 512  # e.g., original fundus usually > 512x512

# Get all image files first
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Process each image with progress bar
for filename in tqdm(image_files, desc="Separating images"):
    file_path = os.path.join(input_dir, filename)
    img = cv2.imread(file_path)

    if img is None:
        print(f"❌ Could not read: {file_path}")
        continue

    h, w, c = img.shape

    # Heuristic: originals are large and RGB; processed are smaller or filtered
    if c == 3 and min(h, w) >= min_size_for_original:
        shutil.copy(file_path, os.path.join(original_dir, filename))
    else:
        shutil.copy(file_path, os.path.join(processed_dir, filename))

print("✅ Separation done.")

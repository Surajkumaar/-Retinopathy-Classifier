import os
import shutil
import pandas as pd
from tqdm import tqdm
labels_df = pd.read_csv(r"D:\archive\test_4.csv")
src_dir = r"D:\archive\test_images_512\test_images_512"
dst_dir = r"D:\archive\total\4"
os.makedirs(dst_dir, exist_ok=True)
image_names = labels_df['image'].values
successful_copies = []
failed_copies = []
missing_in_source = []
for image_name in tqdm(image_names, desc="Copying images", unit="image"):
    src_path = os.path.join(src_dir, f"{image_name}.jpg")
    dst_path = os.path.join(dst_dir, f"{image_name}.jpg")
    
    try:
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            successful_copies.append(image_name)
        else:
            failed_copies.append(image_name)
            missing_in_source.append(image_name)
            tqdm.write(f"Warning: Image not found - {image_name}.jpg")
    except Exception as e:
        failed_copies.append(image_name)
        tqdm.write(f"Error copying {image_name}.jpg: {str(e)}")

print("\nCopy Summary:")
print(f"Total images in CSV: {len(image_names)}")
print(f"Successfully copied: {len(successful_copies)}")
print(f"Failed to copy: {len(failed_copies)}")
print(f"\nMissing Images Summary:")
print(f"Images in CSV but missing from source: {len(missing_in_source)}")

if missing_in_source:
    print("\nList of missing images:")
    for img in missing_in_source:
        print(f"{img}.jpg")


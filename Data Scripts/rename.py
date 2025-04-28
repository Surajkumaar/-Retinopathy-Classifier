import os

# Path to the root directory that contains class folders like '0', '1', ...
root_dir = r"D:\archive (2)\dr_unified_v2\dr_unified_v1\train"

# Starting number (e.g., image_00259 means start from 260)
start_index = 0
counter = start_index

# Loop through class folders (modify if folders are named differently)
for class_folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, class_folder)
    if not os.path.isdir(folder_path):
        continue

    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # Optional for consistency

    for filename in image_files:
        ext = os.path.splitext(filename)[1]
        new_name = f"image_{counter:05d}{ext}"  # image_00260.jpg format
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        counter += 1

    print(f"Processed folder: {class_folder}")

print(f"Renaming complete! Final image number: {counter - 1}")

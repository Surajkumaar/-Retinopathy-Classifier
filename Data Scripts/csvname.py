import os
import csv

# Path to the root folder containing class folders (0, 1, 2, 3, 4)
root_dir =r"D:\archive (2)\dr_unified_v2\dr_unified_v1\train"
csv_path = os.path.join(root_dir, "train_labels.csv")  # Output CSV file

# Prepare list for logging
log = []

# Traverse class folders
for class_label in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, class_label)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            log.append([filename, class_label])

# Write to CSV
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "level"])
    writer.writerows(log)

print(f"CSV log created successfully: {csv_path}")

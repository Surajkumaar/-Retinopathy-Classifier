import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(512, 512)):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    valid_exts = ('.jpg', '.jpeg', '.png')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_exts):
            img_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(img_path)

                # Center-crop to square
                min_side = min(img.size)
                left = (img.width - min_side) // 2
                top = (img.height - min_side) // 2
                right = left + min_side
                bottom = top + min_side
                img_cropped = img.crop((left, top, right, bottom))

                # Resize
                img_resized = img_cropped.resize(size)

                # Save to output folder
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)
                print(f"✅ Processed: {filename}")

            except Exception as e:
                print(f"⚠️ Failed: {filename} — {e}")

    print("🎉 All images processed.")

# === USAGE EXAMPLE ===
resize_images(r"D:\archive (2)\dr_unified_v2\dr_unified_v1\train\balanced dataset\train_balanced", r"D:\archive (2)\dr_unified_v2\dr_unified_v1\train\balanced dataset\resized")

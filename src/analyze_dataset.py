import os
from PIL import Image

DATASET_DIR = "dataset"

splits = ["train", "validation", "test"]
classes = ["NORMAL", "PNEUMONIA"]

print("=" * 50)
print("CHEST X-RAY DATASET ANALYSIS")
print("=" * 50)

total = 0

for split in splits:
    print(f"\n{split.upper()}")

    split_total = 0

    for cls in classes:
        folder = os.path.join(DATASET_DIR, split, cls)

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        ]

        count = len(files)
        split_total += count
        total += count

        print(f"{cls:12} : {count}")

    print(f"{'Total':12} : {split_total}")

print("\n" + "=" * 50)
print(f"TOTAL IMAGES : {total}")
print("=" * 50)

# Check image properties
print("\nIMAGE PROPERTY CHECK")

sample_folder = os.path.join(DATASET_DIR, "train", "NORMAL")
sample_files = [
    f for f in os.listdir(sample_folder)
    if f.lower().endswith((".jpeg", ".jpg", ".png"))
]

for filename in sample_files[:10]:
    path = os.path.join(sample_folder, filename)

    try:
        with Image.open(path) as img:
            print(
                f"{filename:35} "
                f"Size={img.size} "
                f"Mode={img.mode}"
            )
    except Exception as e:
        print(f"Error reading {filename}: {e}")
import zipfile
import os
import shutil

ZIP_FILE = "archive.zip"
OUTPUT_DIR = "dataset"

mapping = {
    "train": "train",
    "test": "test",
    "val": "validation"
}

count = 0

print("Opening archive...")

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    for name in z.namelist():

        parts = name.split("/")

        # Expected:
        # chest_xray/train/NORMAL/image.jpeg
        # chest_xray/train/PNEUMONIA/image.jpeg
        if len(parts) != 4:
            continue

        root, split, category, filename = parts

        if root != "chest_xray":
            continue

        if split not in mapping:
            continue

        if category not in ["NORMAL", "PNEUMONIA"]:
            continue

        if not filename.lower().endswith((".jpeg", ".jpg", ".png")):
            continue

        output_folder = os.path.join(
            OUTPUT_DIR,
            mapping[split],
            category
        )

        os.makedirs(output_folder, exist_ok=True)

        output_file = os.path.join(
            output_folder,
            filename
        )

        with z.open(name) as source:
            with open(output_file, "wb") as target:
                shutil.copyfileobj(source, target)

        count += 1

        if count % 500 == 0:
            print(f"Extracted {count} images...")

print()
print("================================")
print(f"Extraction completed: {count} images")
print("================================")
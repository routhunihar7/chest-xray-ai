import os
import random
import shutil

DATASET_DIR = "dataset"
SEED = 42
VALIDATION_RATIO = 0.10

random.seed(SEED)

classes = ["NORMAL", "PNEUMONIA"]


def get_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ]


print("=" * 55)
print("CREATING PROPER VALIDATION SET")
print("=" * 55)


# ---------------------------------------------------------
# STEP 1: Move existing validation images back to train
# ---------------------------------------------------------

for cls in classes:

    train_dir = os.path.join(DATASET_DIR, "train", cls)
    val_dir = os.path.join(DATASET_DIR, "validation", cls)

    old_validation_images = get_images(val_dir)

    print(
        f"\nMoving existing validation {cls}: "
        f"{len(old_validation_images)} images"
    )

    for filename in old_validation_images:

        source = os.path.join(val_dir, filename)
        destination = os.path.join(train_dir, filename)

        shutil.move(source, destination)


# ---------------------------------------------------------
# STEP 2: Create new validation set
# ---------------------------------------------------------

for cls in classes:

    train_dir = os.path.join(DATASET_DIR, "train", cls)
    val_dir = os.path.join(DATASET_DIR, "validation", cls)

    os.makedirs(val_dir, exist_ok=True)

    images = get_images(train_dir)

    random.shuffle(images)

    validation_count = int(len(images) * VALIDATION_RATIO)

    validation_images = images[:validation_count]

    print(
        f"\n{cls}:"
        f"\n  Total available : {len(images)}"
        f"\n  Moving to validation : {validation_count}"
    )

    for filename in validation_images:

        source = os.path.join(train_dir, filename)
        destination = os.path.join(val_dir, filename)

        shutil.move(source, destination)


# ---------------------------------------------------------
# STEP 3: Final counts
# ---------------------------------------------------------

print("\n" + "=" * 55)
print("FINAL DATASET SPLIT")
print("=" * 55)

total = 0

for split in ["train", "validation", "test"]:

    print(f"\n{split.upper()}")

    split_total = 0

    for cls in classes:

        folder = os.path.join(DATASET_DIR, split, cls)

        count = len(get_images(folder))

        split_total += count
        total += count

        print(f"{cls:12}: {count}")

    print(f"{'Total':12}: {split_total}")


print("\n" + "=" * 55)
print(f"TOTAL IMAGES: {total}")
print("=" * 55)
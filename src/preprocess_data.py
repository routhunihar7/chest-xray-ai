import tensorflow as tf
from tensorflow.keras import layers
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("LOADING CHEST X-RAY DATASET")
print("=" * 60)

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass names:")
print(train_ds.class_names)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="data_augmentation")

# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# CHECK ONE BATCH
# ============================================================

for images, labels in train_ds.take(1):

    print("\nBatch information:")
    print("Image shape :", images.shape)
    print("Label shape :", labels.shape)
    print("Image dtype :", images.dtype)

print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE READY")
print("=" * 60)
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"
MODEL_DIR = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATASET")
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

print("\nClasses:")
print(train_ds.class_names)

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)

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
# EFFICIENTNETB0
# ============================================================

print("\nLoading EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

print("EfficientNetB0 loaded.")
print("Base model frozen.")

# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nMODEL SUMMARY")

model.summary()

# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_counts = {
    0: 1215,
    1: 3495
}

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.repeat(
        classes,
        [class_counts[0], class_counts[1]]
    )
)

class_weights = {
    0: float(weights[0]),
    1: float(weights[1])
}

print("Class weights:")
print(class_weights)

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            MODEL_DIR,
            "efficientnet_best.keras"
        ),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "efficientnet_final.keras"
)

model.save(final_model_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Final model saved to:")
print(final_model_path)
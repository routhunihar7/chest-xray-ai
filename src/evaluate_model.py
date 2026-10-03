import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"
MODEL_PATH = "models/efficientnet_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
print(class_names)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

results = model.evaluate(
    test_ds,
    verbose=1
)

print("\nRaw Keras test results:")

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_probability = []

for images, labels in test_ds:

    probabilities = model.predict(
        images,
        verbose=0
    )

    y_probability.extend(
        probabilities.ravel()
    )

    y_true.extend(
        labels.numpy().ravel()
    )


y_true = np.array(y_true).astype(int)
y_probability = np.array(y_probability)

# Convert probability to class
y_pred = (y_probability >= 0.5).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_probability
)


print("\n" + "=" * 60)
print("FINAL TEST METRICS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("Confusion Matrix - EfficientNetB0")

plt.colorbar()

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    "results/evaluation_results.txt",
    "w"
) as f:

    f.write("EfficientNetB0 Test Evaluation\n")
    f.write("=" * 40 + "\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n")
    f.write(f"ROC-AUC   : {auc:.4f}\n")

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nEvaluation completed successfully.")

print("\nSaved:")
print("results/confusion_matrix.png")
print("results/evaluation_results.txt")
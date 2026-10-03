import os
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = "dataset"

classes = ["NORMAL", "PNEUMONIA"]

fig, axes = plt.subplots(2, 4, figsize=(12, 7))

for row, cls in enumerate(classes):

    folder = os.path.join(
        DATASET_DIR,
        "train",
        cls
    )

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ]

    selected = random.sample(images, 4)

    for col, filename in enumerate(selected):

        path = os.path.join(folder, filename)

        image = Image.open(path)

        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(
                cls,
                fontsize=14
            )

plt.suptitle(
    "Chest X-Ray Dataset: Normal vs Pneumonia",
    fontsize=16
)

plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/dataset_visualization.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

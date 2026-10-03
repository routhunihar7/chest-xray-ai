import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_best.keras"
IMG_SIZE = (224, 224)

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# GET MODEL COMPONENTS
# ============================================================

efficientnet = model.get_layer("efficientnetb0")

gap_layer = model.get_layer(
    "global_average_pooling2d"
)

dropout_layer = model.get_layer(
    "dropout"
)

dense_layer = model.get_layer(
    "dense"
)

print("\nEfficientNetB0 found.")

# ============================================================
# FIND LAST 4D FEATURE LAYER
# ============================================================

last_conv_layer = None

for layer in reversed(efficientnet.layers):

    try:

        shape = layer.output.shape

        if len(shape) == 4:

            last_conv_layer = layer
            break

    except Exception:
        pass


if last_conv_layer is None:

    raise RuntimeError(
        "Could not find the final convolutional feature layer."
    )


print(
    "Grad-CAM layer:",
    last_conv_layer.name
)

# ============================================================
# CREATE INTERNAL EFFICIENTNET MODEL
# ============================================================

feature_model = tf.keras.models.Model(
    inputs=efficientnet.input,
    outputs=[
        last_conv_layer.output,
        efficientnet.output
    ]
)

print("Grad-CAM feature model created.")

# ============================================================
# FIND TEST IMAGE
# ============================================================

pneumonia_dir = "dataset/test/PNEUMONIA"

pneumonia_images = [
    os.path.join(pneumonia_dir, f)
    for f in os.listdir(pneumonia_dir)
    if f.lower().endswith(
        (".jpeg", ".jpg", ".png")
    )
]

if len(pneumonia_images) == 0:

    raise RuntimeError(
        "No test pneumonia images found."
    )

IMAGE_PATH = pneumonia_images[0]

print("\nUsing image:")
print(IMAGE_PATH)

# ============================================================
# LOAD IMAGE
# ============================================================

image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE,
    color_mode="rgb"
)

image_array = tf.keras.utils.img_to_array(
    image
)

input_image = np.expand_dims(
    image_array,
    axis=0
)

input_image = tf.cast(
    input_image,
    tf.float32
)

# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    input_image,
    verbose=0
)[0][0]

if prediction >= 0.5:

    predicted_class = "PNEUMONIA"
    confidence = prediction

else:

    predicted_class = "NORMAL"
    confidence = 1 - prediction


print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)

print(
    "Predicted class:",
    predicted_class
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)

# ============================================================
# GRAD-CAM
# ============================================================

print("\nGenerating Grad-CAM...")

with tf.GradientTape() as tape:

    # Get EfficientNet features
    conv_outputs, efficientnet_output = feature_model(
        input_image,
        training=False
    )

    # Classification head
    x = gap_layer(
        efficientnet_output
    )

    x = dropout_layer(
        x,
        training=False
    )

    predictions = dense_layer(
        x
    )

    class_score = predictions[:, 0]


# ============================================================
# CALCULATE GRADIENTS
# ============================================================

gradients = tape.gradient(
    class_score,
    conv_outputs
)

# Average gradients across height and width

pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(1, 2)
)

conv_outputs = conv_outputs[0]

pooled_gradients = pooled_gradients[0]

# Weight feature maps

heatmap = tf.reduce_sum(
    conv_outputs * pooled_gradients,
    axis=-1
)

# ReLU

heatmap = tf.maximum(
    heatmap,
    0
)

# Normalize

max_value = tf.reduce_max(
    heatmap
)

if float(max_value) > 0:

    heatmap = heatmap / max_value

heatmap = heatmap.numpy()

# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

# ============================================================
# SAVE RAW HEATMAP
# ============================================================

heatmap_path = (
    "results/gradcam_heatmap.png"
)

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.colorbar()

plt.title(
    f"Grad-CAM - {predicted_class}"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    heatmap_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# CREATE OVERLAY
# ============================================================

original_image = tf.keras.utils.img_to_array(
    tf.keras.utils.load_img(
        IMAGE_PATH,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )
)

# Resize heatmap

heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    IMG_SIZE
)

heatmap_resized = tf.squeeze(
    heatmap_resized
).numpy()

# Convert to 0-255

heatmap_uint8 = np.uint8(
    255 * heatmap_resized
)

# Convert to RGB color map

heatmap_color = plt.cm.jet(
    heatmap_uint8
)

heatmap_color = np.uint8(
    heatmap_color[:, :, :3] * 255
)

# Overlay original image + heatmap

overlay = (
    0.6 * original_image +
    0.4 * heatmap_color
)

overlay = np.clip(
    overlay,
    0,
    255
).astype(np.uint8)

# ============================================================
# SAVE OVERLAY
# ============================================================

overlay_path = (
    "results/gradcam_overlay.png"
)

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    overlay
)

plt.title(
    f"Grad-CAM Explanation\n"
    f"{predicted_class} - "
    f"{confidence * 100:.2f}%"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    overlay_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("GRAD-CAM COMPLETE")
print("=" * 60)

print(
    "Saved:",
    heatmap_path
)

print(
    "Saved:",
    overlay_path
)
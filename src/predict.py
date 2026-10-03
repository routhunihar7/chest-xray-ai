import tensorflow as tf
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_best.keras"

IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image):

    """
    Predict NORMAL or PNEUMONIA
    from a chest X-ray image.
    """

    # Resize
    image = image.resize(
        IMG_SIZE
    )

    # Convert image to array
    image_array = np.array(
        image
    )

    # Make sure image has 3 channels
    if len(image_array.shape) == 2:

        image_array = np.stack(
            [image_array] * 3,
            axis=-1
        )

    elif image_array.shape[-1] == 4:

        image_array = image_array[:, :, :3]

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Classification
    if prediction >= 0.5:

        label = "PNEUMONIA"
        confidence = float(prediction)

    else:

        label = "NORMAL"
        confidence = float(1 - prediction)

    return label, confidence
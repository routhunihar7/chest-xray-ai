import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_best.keras"
IMG_SIZE = (224, 224)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #333333;
    }

    .normal {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #333333;
    }

    .metric {
        font-size: 30px;
        font-weight: bold;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(model, image):

    image = image.convert("RGB")

    resized = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        resized
    ).astype(
        np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    if prediction >= 0.5:

        label = "PNEUMONIA"
        confidence = float(prediction)

    else:

        label = "NORMAL"
        confidence = float(1 - prediction)

    return label, confidence


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(model, image):

    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    original_image = np.array(
        image
    )

    # --------------------------------------------------------
    # Resize input
    # --------------------------------------------------------

    resized_image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        resized_image
    ).astype(
        np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Get nested EfficientNetB0 model
    # --------------------------------------------------------

    efficientnet = model.get_layer(
        name="efficientnetb0"
    )

    # --------------------------------------------------------
    # Get target Grad-CAM layer
    # --------------------------------------------------------

    target_layer = efficientnet.get_layer(
        name="top_activation"
    )

    # --------------------------------------------------------
    # Create feature model
    #
    # This model takes EfficientNet input and returns:
    # 1. top_activation feature maps
    # 2. EfficientNet output
    # --------------------------------------------------------

    feature_model = tf.keras.models.Model(
        inputs=efficientnet.input,
        outputs=[
            target_layer.output,
            efficientnet.output
        ]
    )

    # --------------------------------------------------------
    # Get data augmentation layer
    # --------------------------------------------------------

    augmentation = model.get_layer(
        name="data_augmentation"
    )

    # --------------------------------------------------------
    # Get classification layers
    # --------------------------------------------------------

    pooling = model.get_layer(
        name="global_average_pooling2d"
    )

    dropout = model.get_layer(
        name="dropout"
    )

    classifier = model.get_layer(
        name="dense"
    )

    # --------------------------------------------------------
    # Forward pass through the same pipeline
    # --------------------------------------------------------

    augmented_image = augmentation(
        image_array,
        training=False
    )

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # EfficientNet feature maps + output
        conv_features, efficientnet_output = feature_model(
            augmented_image,
            training=False
        )

        # Classification head
        x = pooling(
            efficientnet_output
        )

        x = dropout(
            x,
            training=False
        )

        prediction = classifier(
            x
        )

        # Pneumonia score
        class_score = prediction[:, 0]

    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_features
    )

    # Safety check
    if gradients is None:

        raise RuntimeError(
            "Gradients could not be calculated."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_features = conv_features[0]

    pooled_gradients = pooled_gradients[0]

    # --------------------------------------------------------
    # Weight feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_features * pooled_gradients,
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

    heatmap = heatmap / (
        max_value + 1e-8
    )

    heatmap = heatmap.numpy()

    # --------------------------------------------------------
    # Resize heatmap to original image size
    # --------------------------------------------------------

    height, width = original_image.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (width, height)
    )

    # --------------------------------------------------------
    # Convert heatmap to color
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_image,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    return Image.fromarray(
        overlay
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🫁 Chest X-Ray AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Pneumonia Classification'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Model")

    st.write(
        """
        This application uses **EfficientNetB0**
        with transfer learning to classify
        chest X-ray images into:

        - NORMAL
        - PNEUMONIA
        """
    )

    st.divider()

    st.write("**Model:** EfficientNetB0")
    st.write("**Learning:** Transfer Learning")
    st.write("**Input:** Chest X-ray")
    st.write("**Image Size:** 224 × 224")
    st.write("**Task:** Binary Classification")
    st.write("**Explainability:** Grad-CAM")

    st.divider()

    st.caption(
        "For research and educational purposes only. "
        "This tool is not a medical diagnosis."
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("Upload Chest X-Ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # IMAGE + PREDICTION
    # --------------------------------------------------------

    col1, col2 = st.columns(
        [1.1, 0.9]
    )

    with col1:

        st.subheader(
            "Uploaded X-Ray"
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "AI Prediction"
        )

        with st.spinner(
            "Analyzing X-ray..."
        ):

            label, confidence = predict_image(
                model,
                image
            )

        confidence_percentage = (
            confidence * 100
        )

        if label == "PNEUMONIA":

            st.markdown(
                f"""
                <div class="result-box warning">

                <h2>⚠️ PNEUMONIA</h2>

                <p class="metric">
                {confidence_percentage:.2f}%
                </p>

                <p>
                Model confidence
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-box normal">

                <h2>✅ NORMAL</h2>

                <p class="metric">
                {confidence_percentage:.2f}%
                </p>

                <p>
                Model confidence
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Prediction Confidence"
    )

    st.progress(
        confidence
    )

    st.caption(
        f"{confidence_percentage:.2f}% confidence for "
        f"{label}"
    )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🧠 Explainable AI - Grad-CAM"
    )

    st.write(
        """
        Grad-CAM provides a visual explanation of the
        regions that contributed to the model's prediction.
        """
    )

    try:

        with st.spinner(
            "Generating Grad-CAM explanation..."
        ):

            gradcam_image = generate_gradcam(
                model,
                image
            )

        grad_col1, grad_col2 = st.columns(
            2
        )

        with grad_col1:

            st.markdown(
                "**Original X-Ray**"
            )

            st.image(
                image,
                use_container_width=True
            )

        with grad_col2:

            st.markdown(
                "**Grad-CAM Explanation**"
            )

            st.image(
                gradcam_image,
                use_container_width=True
            )

        st.info(
            "The highlighted regions indicate areas that "
            "contributed more strongly to the model's "
            "prediction. Grad-CAM is an explanatory "
            "visualization and is not a clinical diagnosis."
        )

    except Exception as e:

        st.error(
            "Grad-CAM could not be generated."
        )

        st.exception(e)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader(
    "📊 Model Performance"
)

metric1, metric2, metric3, metric4, metric5 = st.columns(
    5
)

with metric1:

    st.metric(
        "Accuracy",
        "86.38%"
    )

with metric2:

    st.metric(
        "Precision",
        "84.27%"
    )

with metric3:

    st.metric(
        "Recall",
        "96.15%"
    )

with metric4:

    st.metric(
        "F1 Score",
        "89.82%"
    )

with metric5:

    st.metric(
        "ROC-AUC",
        "94.82%"
    )
st.divider()

st.header("📊 Confusion Matrix")

st.write(
    "The confusion matrix shows how the model classified the 624 images "
    "in the independent test set."
)

cm = np.array([
    [164, 70],
    [15, 375]
])

fig, ax = plt.subplots(figsize=(7, 5))

im = ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
ax.set_yticklabels(["NORMAL", "PNEUMONIA"])

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=18
        )

plt.colorbar(im, ax=ax)

st.pyplot(fig)
plt.close(fig)

# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.subheader(
    "📁 Dataset Information"
)

data1, data2, data3, data4 = st.columns(
    4
)

with data1:

    st.metric(
        "Training",
        "4,710"
    )

with data2:

    st.metric(
        "Validation",
        "522"
    )

with data3:

    st.metric(
        "Test",
        "624"
    )

with data4:

    st.metric(
        "Total",
        "5,856"
    )


# ============================================================
# SYSTEM WORKFLOW
# ============================================================

st.divider()

st.subheader(
    "🔄 How the System Works"
)

step1, step2, step3, step4, step5 = st.columns(
    5
)

with step1:

    st.markdown(
        """
        ### 1️⃣ Upload

        Upload a chest
        X-ray image.
        """
    )

with step2:

    st.markdown(
        """
        ### 2️⃣ Preprocess

        Resize image
        to 224 × 224.
        """
    )

with step3:

    st.markdown(
        """
        ### 3️⃣ AI Model

        EfficientNetB0
        analyzes the image.
        """
    )

with step4:

    st.markdown(
        """
        ### 4️⃣ Prediction

        NORMAL or
        PNEUMONIA.
        """
    )

with step5:

    st.markdown(
        """
        ### 5️⃣ Explain

        Grad-CAM
        visualizes model focus.
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ This application is an AI research prototype "
    "for educational purposes. It should not be used "
    "as a substitute for professional medical diagnosis."
)
st.divider()

st.warning(
    "⚠️ Research / Educational Prototype: "
    "This system is developed for academic and research purposes. "
    "It is not a clinical diagnostic system and should not be used "
    "to make medical decisions. Model predictions and Grad-CAM "
    "visualizations require validation by qualified medical professionals."
)
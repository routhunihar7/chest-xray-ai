import streamlit as st
from PIL import Image

from src.predict import predict_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🫁",
    layout="wide"
)


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
        color: #666;
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
    }

    .normal {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }

    .metric {
        font-size: 30px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
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
    'Deep Learning based Pneumonia Classification'
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
    st.write("**Input:** Chest X-ray")
    st.write("**Image Size:** 224 × 224")
    st.write("**Task:** Binary Classification")

    st.divider()

    st.caption(
        "For research and educational purposes only. "
        "This tool is not a medical diagnosis."
    )


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
# DISPLAY AND PREDICT
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    col1, col2 = st.columns(
        2
    )

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Uploaded X-Ray"
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "AI Prediction"
        )

        with st.spinner(
            "Analyzing X-ray..."
        ):

            label, confidence = predict_image(
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


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.subheader(
    "How the System Works"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        ### 1️⃣ Upload

        Upload a chest
        X-ray image.
        """
    )

with col2:

    st.markdown(
        """
        ### 2️⃣ Preprocess

        Image is resized
        to 224 × 224.
        """
    )

with col3:

    st.markdown(
        """
        ### 3️⃣ AI Model

        EfficientNetB0
        analyzes the image.
        """
    )

with col4:

    st.markdown(
        """
        ### 4️⃣ Result

        NORMAL or
        PNEUMONIA.
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
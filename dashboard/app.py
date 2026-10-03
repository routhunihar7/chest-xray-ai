import os
import sys
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is in sys.path for modular imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.health_guidance import (
    COMMON_SYMPTOMS,
    get_health_guidance,
    get_emergency_warning,
    get_medical_disclaimer,
    get_gradcam_disclaimer,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chest X-Ray AI — Pulmonary Intelligence",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_best.keras"
if not os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(BASE_DIR, MODEL_PATH)):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

IMG_SIZE = (224, 224)


# ============================================================
# HELPER TO RENDER CLEAN HTML (AVOIDS MARKDOWN PRE/CODE BLOCKS)
# ============================================================

def render_html(html_str: str):
    """
    Renders HTML safely in Streamlit, ensuring no leading indentation
    triggers Markdown code block parsing.
    """
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# ULTRA-MODERN DYNAMIC CSS & ANIMATIONS
# ============================================================

render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, h4, .main-title, .section-title {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Keyframe Animations */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes pulseGlowRed {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes floatAnimation {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Header Styling */
    .hero-container {
        text-align: center;
        padding: 26px 20px 20px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(99, 102, 241, 0.08) 50%, rgba(236, 72, 153, 0.08) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 25px;
        animation: fadeInUp 0.7s ease-out;
    }

    .hero-badge {
        display: inline-block;
        padding: 5px 15px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-radius: 30px;
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        margin-bottom: 10px;
        animation: floatAnimation 3s ease-in-out infinite;
    }

    .main-title {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 50%, #db2777 100%);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }

    .subtitle {
        font-size: 17px;
        color: #475569;
        font-weight: 500;
        max-width: 700px;
        margin: 0 auto;
    }

    /* Result Cards */
    .result-box-animated {
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
        border-width: 2px;
        border-style: solid;
        transition: transform 0.3s ease;
    }

    .result-box-animated:hover {
        transform: scale(1.01);
    }

    .result-normal-animated {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #22c55e;
        box-shadow: 0 10px 24px -4px rgba(34, 197, 94, 0.25);
    }

    .result-pneumonia-animated {
        background: linear-gradient(135deg, #fffbeb 0%, #fee2e2 100%);
        border-color: #ef4444;
        box-shadow: 0 10px 24px -4px rgba(239, 68, 68, 0.25);
        animation: pulseGlowRed 3s infinite;
    }

    .card-top-tag {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .result-title-normal {
        font-size: 30px;
        font-weight: 900;
        color: #15803d;
        margin: 6px 0;
    }

    .result-title-pneumonia {
        font-size: 30px;
        font-weight: 900;
        color: #b91c1c;
        margin: 6px 0;
    }

    .metric-animated {
        font-size: 38px;
        font-weight: 900;
        color: #0f172a;
        margin: 2px 0;
        font-family: 'Outfit', sans-serif;
    }

    .proto-disclaimer {
        font-size: 12.5px;
        color: #64748b;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px dashed rgba(100, 116, 139, 0.3);
        font-style: italic;
    }

    /* Health Guidance Grid */
    .guidance-section-wrap {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
        margin-top: 15px;
        animation: fadeInUp 0.6s ease-out;
    }

    .guidance-banner-normal {
        background: linear-gradient(90deg, #dcfce7 0%, #e0f2fe 100%);
        border-left: 6px solid #10b981;
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 16px;
        color: #065f46;
    }

    .guidance-banner-pneumonia {
        background: linear-gradient(90deg, #fef3c7 0%, #fee2e2 100%);
        border-left: 6px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 16px;
        color: #92400e;
    }

    .guidance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 14px;
        margin-top: 16px;
        margin-bottom: 16px;
    }

    .tip-card {
        background: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: all 0.25s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }

    .tip-card:hover {
        background: #ffffff;
        border-color: #0ea5e9;
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(14, 165, 233, 0.12);
    }

    .tip-icon-box {
        font-size: 22px;
        min-width: 42px;
        height: 42px;
        border-radius: 10px;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
    }

    .tip-title {
        font-size: 14.5px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }

    .tip-text {
        font-size: 13px;
        color: #475569;
        line-height: 1.45;
        margin: 0;
    }

    /* Emergency Warning Card */
    .emergency-card-animated {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 10px 26px rgba(244, 63, 94, 0.12);
        margin-top: 20px;
        margin-bottom: 20px;
        animation: pulseGlowRed 3.5s infinite;
    }

    .emergency-badge-pulse {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #e11d48;
        color: #ffffff;
        font-weight: 800;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 5px 12px;
        border-radius: 20px;
        box-shadow: 0 3px 10px rgba(225, 29, 72, 0.35);
        margin-bottom: 10px;
    }

    .emergency-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 10px;
        margin: 14px 0;
    }

    .emergency-item {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #fecdd3;
        border-radius: 8px;
        padding: 9px 12px;
        font-size: 13px;
        font-weight: 600;
        color: #9f1239;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .emergency-statement-box {
        background: #e11d48;
        color: #ffffff;
        font-weight: 700;
        font-size: 14px;
        padding: 12px 18px;
        border-radius: 10px;
        margin-top: 12px;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.25);
        text-align: center;
    }

    /* Doctor Follow-Up Box */
    .followup-card-animated {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 2px solid #38bdf8;
        border-radius: 16px;
        padding: 18px 22px;
        margin-top: 16px;
        box-shadow: 0 6px 18px rgba(56, 189, 248, 0.1);
    }

    .symptom-tag-pill {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 4px 6px 4px 0;
        box-shadow: 0 2px 6px rgba(99, 102, 241, 0.2);
    }

    /* Metric Cards */
    .metric-card-gradient {
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        color: #ffffff;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .metric-card-gradient:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
    }

    .grad-1 { background: linear-gradient(135deg, #0284c7, #0369a1); }
    .grad-2 { background: linear-gradient(135deg, #6366f1, #4338ca); }
    .grad-3 { background: linear-gradient(135deg, #10b981, #047857); }
    .grad-4 { background: linear-gradient(135deg, #8b5cf6, #6d28d9); }
    .grad-5 { background: linear-gradient(135deg, #f59e0b, #d97706); }

    .data-metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        transition: all 0.2s ease;
    }

    .data-metric-card:hover {
        background: #ffffff;
        border-color: #6366f1;
        transform: translateY(-2px);
    }

    /* Workflow Step Cards */
    .workflow-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #0ea5e9;
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        transition: all 0.25s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }

    .workflow-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(14, 165, 233, 0.12);
    }

    .step-1 { border-top-color: #0ea5e9; }
    .step-2 { border-top-color: #6366f1; }
    .step-3 { border-top-color: #a855f7; }
    .step-4 { border-top-color: #ec4899; }
    .step-5 { border-top-color: #10b981; }

    /* Footer Disclaimer */
    .disclaimer-card-footer {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border: 1px solid #facc15;
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 30px;
        color: #713f12;
        font-size: 13.5px;
        line-height: 1.6;
        box-shadow: 0 4px 14px rgba(250, 204, 21, 0.12);
    }
    </style>
    """
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

        conv_features, efficientnet_output = feature_model(
            augmented_image,
            training=False
        )

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

        class_score = prediction[:, 0]

    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_features
    )

    if gradients is None:
        raise RuntimeError("Gradients could not be calculated.")

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_features = conv_features[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_features * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to original image size
    height, width = original_image.shape[:2]
    heatmap = cv2.resize(heatmap, (width, height))

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_image,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    return Image.fromarray(overlay)


# ============================================================
# MODULAR HEALTH GUIDANCE & UI RENDERING FUNCTIONS
# ============================================================

def render_optional_symptom_input():
    """
    Renders an optional symptom questionnaire in a beautiful card.
    Symptoms are never used to alter the AI prediction or produce a medical diagnosis.
    """
    symptom_box_html = """
<div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(14, 165, 233, 0.05) 100%); border: 1px solid #e0e7ff; border-radius: 16px; padding: 18px 22px; margin-bottom: 18px;">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <span style="font-size: 22px;">📋</span>
        <span style="font-size: 18px; font-weight: 700; color: #1e1b4b;">Patient Information & Symptoms (Optional)</span>
    </div>
    <p style="font-size: 13.5px; color: #64748b; margin: 0;">
        ℹ️ Select any current symptoms to prepare discussion notes for your doctor.
        <strong>These symptoms are collected only for educational guidance and are not used to diagnose a condition.</strong>
    </p>
</div>
"""
    render_html(symptom_box_html)

    col1, col2, col3 = st.columns(3)

    with col1:
        s_cough = st.checkbox("🫁 Cough", key="sym_cough")
        s_fever = st.checkbox("🌡️ Fever", key="sym_fever")

    with col2:
        s_sob = st.checkbox("💨 Shortness of breath", key="sym_sob")
        s_chest = st.checkbox("💔 Chest discomfort", key="sym_chest")

    with col3:
        s_fatigue = st.checkbox("🥱 Fatigue", key="sym_fatigue")
        s_none = st.checkbox("✅ None of these", key="sym_none")

    selected_symptoms = []
    if not s_none:
        if s_cough:
            selected_symptoms.append("Cough")
        if s_fever:
            selected_symptoms.append("Fever")
        if s_sob:
            selected_symptoms.append("Shortness of breath")
        if s_chest:
            selected_symptoms.append("Chest discomfort")
        if s_fatigue:
            selected_symptoms.append("Fatigue")
    else:
        selected_symptoms.append("None of these")

    return selected_symptoms


def render_patient_friendly_result(label: str, confidence: float):
    """
    Renders the dynamic, animated patient-friendly AI X-Ray Analysis card.
    """
    confidence_percentage = confidence * 100

    if label == "PNEUMONIA":
        box_class = "result-pneumonia-animated"
        title_html = '<div class="result-title-pneumonia">⚠️ PNEUMONIA</div>'
        badge_text = "AI MODEL CLASSIFICATION: PNEUMONIA"
        badge_color = "#b91c1c"
    else:
        box_class = "result-normal-animated"
        title_html = '<div class="result-title-normal">✅ NORMAL</div>'
        badge_text = "AI MODEL CLASSIFICATION: NORMAL"
        badge_color = "#15803d"

    result_html = f"""
<div class="result-box-animated {box_class}">
    <div class="card-top-tag" style="color: {badge_color};">{badge_text}</div>
    {title_html}
    <div class="metric-animated">{confidence_percentage:.2f}%</div>
    <div style="font-size: 13.5px; font-weight: 600; color: #475569;">Model Confidence</div>
    <p class="proto-disclaimer">
        Important: This result is generated by an AI research prototype and should not be considered a medical diagnosis.
    </p>
</div>
"""
    render_html(result_html)


def render_health_guidance(prediction: str, selected_symptoms=None):
    """
    Renders the colorful, animated AI Health Guidance section after image evaluation.
    Clearly separated visually from the AI model result.
    """
    guidance = get_health_guidance(prediction, selected_symptoms)
    is_pneumonia = (prediction.upper() == "PNEUMONIA")

    banner_class = "guidance-banner-pneumonia" if is_pneumonia else "guidance-banner-normal"
    header_color = "#b45309" if is_pneumonia else "#047857"

    # Build Tip Cards Grid
    cards_html = ""
    for pt in guidance["structured_points"]:
        cards_html += (
            f'<div class="tip-card">'
            f'<div class="tip-icon-box">{pt["icon"]}</div>'
            f'<div>'
            f'<div class="tip-title">{pt["title"]}</div>'
            f'<div class="tip-text">{pt["text"]}</div>'
            f'</div>'
            f'</div>'
        )

    guidance_html = f"""
<div class="guidance-section-wrap">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 12px;">
        <div>
            <h3 style="margin: 0; color: #0f172a; font-size: 24px; font-weight: 800;">
                🩺 General Health Guidance
            </h3>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 14px;">
                Educational and supportive health information. Not a substitute for clinical advice.
            </p>
        </div>
    </div>
    <div class="{banner_class}">
        <div style="font-size: 16px; font-weight: 800; margin-bottom: 4px;">
            {guidance["classification_badge"]}
        </div>
        <div style="font-size: 14px; line-height: 1.5;">
            {guidance["classification_explanation"]}
        </div>
    </div>
    <div style="font-size: 18px; font-weight: 800; color: {header_color}; margin-top: 15px;">
        💡 {guidance["guidance_title"]}
    </div>
    <div class="guidance-grid">
        {cards_html}
    </div>
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; font-size: 13px; color: #475569;">
        {guidance["guidance_note"]}
    </div>
</div>
"""
    render_html(guidance_html)

    # Doctor Follow-up card
    symptoms_html = ""
    if guidance["reported_symptoms"]:
        pills = "".join([f'<span class="symptom-tag-pill">{s}</span>' for s in guidance["reported_symptoms"]])
        symptoms_html = (
            f'<div style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed #cbd5e1;">'
            f'<div style="font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 6px;">'
            f'Selected Symptoms to Discuss with Doctor:'
            f'</div>'
            f'<div>{pills}</div>'
            f'<div style="font-size: 12px; color: #64748b; margin-top: 6px;">'
            f'<em>{guidance["symptom_disclaimer"]}</em>'
            f'</div>'
            f'</div>'
        )

    followup_html = f"""
<div class="followup-card-animated">
    <div style="display: flex; align-items: center; gap: 8px; font-size: 18px; font-weight: 800; color: #0369a1; margin-bottom: 6px;">
        <span>👨‍⚕️</span>
        <span>Recommended Medical Follow-Up</span>
    </div>
    <p style="font-size: 14.5px; color: #1e293b; margin: 0; line-height: 1.5;">
        <strong>{guidance["medical_followup"]}</strong> Please discuss this result and any symptoms with a qualified healthcare professional.
    </p>
    {symptoms_html}
</div>
"""
    render_html(followup_html)


def render_emergency_warning():
    """
    Renders high-visibility animated emergency red flag callout.
    """
    emergency_info = get_emergency_warning()
    items_html = "".join([f'<div class="emergency-item"><span>⚠️</span> {s}</div>' for s in emergency_info["symptoms"]])

    emergency_html = f"""
<div class="emergency-card-animated">
    <div class="emergency-badge-pulse">
        <span>🚨</span>
        <span>URGENT MEDICAL ADVISORY</span>
    </div>
    <h3 style="color: #9f1239; font-size: 22px; font-weight: 800; margin: 6px 0 4px 0;">
        {emergency_info["title"]}
    </h3>
    <p style="color: #9f1239; font-size: 14px; margin-bottom: 12px; font-weight: 500;">
        {emergency_info["intro"]}
    </p>
    <div class="emergency-grid">
        {items_html}
    </div>
    <div class="emergency-statement-box">
        {emergency_info["statement"]}
    </div>
</div>
"""
    render_html(emergency_html)


def render_medical_disclaimer():
    """
    Renders the bottom application disclaimer.
    """
    disclaimer = get_medical_disclaimer()
    disclaimer_html = f"""
<div class="disclaimer-card-footer">
    <div style="display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 16px; margin-bottom: 4px;">
        <span>⚠️</span>
        <span>{disclaimer["title"]}</span>
    </div>
    <p style="margin: 0; font-size: 13.5px; line-height: 1.6;">
        {disclaimer["text"]}
    </p>
</div>
"""
    render_html(disclaimer_html)


# ============================================================
# HERO HEADER
# ============================================================

render_html(
    """
    <div class="hero-container">
        <div class="hero-badge">Deep Learning & Explainable AI</div>
        <div class="main-title">🫁 Chest X-Ray AI</div>
        <div class="subtitle">
            Automated Pulmonary Pneumonia Classification with Explainable Grad-CAM and Safe AI Health Guidance.
        </div>
    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div style="text-align: center; padding: 10px 0 15px 0;">
            <div style="font-size: 32px;">🫁</div>
            <h2 style="font-size: 20px; font-weight: 800; margin: 0; color: #0284c7;">Chest X-Ray AI</h2>
            <span style="font-size: 12px; color: #64748b;">EfficientNetB0 Architecture</span>
        </div>
        """
    )

    st.write(
        """
        This application utilizes **EfficientNetB0** transfer learning to classify
        chest radiograph images into:

        - 🟢 **NORMAL**
        - 🔴 **PNEUMONIA**
        """
    )

    st.divider()

    st.markdown("### ⚙️ System Specifications")
    st.markdown("- **Architecture:** `EfficientNetB0`")
    st.markdown("- **Input Resolution:** `224 × 224 RGB`")
    st.markdown("- **Explainability:** `Grad-CAM Heatmap`")
    st.markdown("- **Feature:** `AI Health Guidance`")

    st.divider()

    st.caption(
        "🛡️ **Academic & Research Prototype:** "
        "Not a diagnostic medical device. Consult a licensed physician for clinical interpretation."
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
# OPTIONAL PATIENT INFORMATION / SYMPTOMS
# ============================================================

selected_symptoms = render_optional_symptom_input()


# ============================================================
# FILE UPLOAD
# ============================================================

render_html(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-top: 15px; margin-bottom: 10px;">
        <span style="font-size: 24px;">📤</span>
        <h2 style="font-size: 22px; font-weight: 800; margin: 0; color: #0f172a;">Upload Chest X-Ray</h2>
    </div>
    """
)

uploaded_file = st.file_uploader(
    "Upload X-ray image (JPEG / PNG)",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# ANALYSIS (DYNAMICALLY GIVING HEALTH TIPS AFTER EVALUATION)
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

        render_html(
            """
            <div style="font-size: 16px; font-weight: 700; color: #334155; margin-bottom: 8px;">
                🖼️ Uploaded Patient Radiograph
            </div>
            """
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        render_html(
            """
            <div style="font-size: 16px; font-weight: 700; color: #334155; margin-bottom: 8px;">
                ⚡ Deep Learning Analysis
            </div>
            """
        )

        with st.spinner(
            "🧠 Neural network evaluating radiograph..."
        ):

            label, confidence = predict_image(
                model,
                image
            )

        # Render patient-friendly AI result card
        render_patient_friendly_result(label, confidence)

    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    confidence_percentage = confidence * 100
    st.divider()

    render_html(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 16px; font-weight: 700; color: #0f172a;">Classification Confidence</span>
            <span style="font-size: 16px; font-weight: 800; color: #0284c7;">{confidence_percentage:.2f}% ({label})</span>
        </div>
        """
    )

    st.progress(
        confidence
    )

    # --------------------------------------------------------
    # GRAD-CAM EXPLAINABILITY
    # --------------------------------------------------------

    st.divider()

    render_html(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            <span style="font-size: 24px;">🧠</span>
            <h3 style="font-size: 22px; font-weight: 800; margin: 0; color: #0f172a;">
                Explainable AI — Grad-CAM Heatmap
            </h3>
        </div>
        <p style="font-size: 14px; color: #64748b; margin-bottom: 16px;">
            Gradient-weighted Class Activation Mapping reveals specific image regions influencing the EfficientNetB0 prediction.
        </p>
        """
    )

    try:

        with st.spinner(
            "Generating gradient class activation maps..."
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
                "**Original Radiograph**"
            )

            st.image(
                image,
                use_container_width=True
            )

        with grad_col2:

            st.markdown(
                "**Grad-CAM Salience Overlay**"
            )

            st.image(
                gradcam_image,
                use_container_width=True
            )

        # Grad-CAM disclaimer (Feature 6)
        render_html(
            f"""
            <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 10px; padding: 12px 16px; font-size: 13px; color: #0369a1; margin-top: 12px;">
                ℹ️ <strong>Grad-CAM Note:</strong> {get_gradcam_disclaimer()}
            </div>
            """
        )

    except Exception as e:

        st.error(
            "Grad-CAM could not be generated."
        )

        st.exception(e)

    # --------------------------------------------------------
    # AI HEALTH GUIDANCE (DYNAMICALLY GIVEN AFTER EVALUATION)
    # --------------------------------------------------------

    st.divider()
    render_health_guidance(label, selected_symptoms)
    render_emergency_warning()


# ============================================================
# MODEL PERFORMANCE (COLORFUL GRADIENT CARDS)
# ============================================================

st.divider()

render_html(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span style="font-size: 24px;">📊</span>
        <h2 style="font-size: 22px; font-weight: 800; margin: 0; color: #0f172a;">Model Performance Metrics</h2>
    </div>
    """
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    render_html(
        """
        <div class="metric-card-gradient grad-1">
            <div style="font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; opacity: 0.9;">Accuracy</div>
            <div style="font-size: 30px; font-weight: 900; margin: 4px 0; font-family: 'Outfit';">86.38%</div>
            <div style="font-size: 11px; opacity: 0.85;">Test Benchmark</div>
        </div>
        """
    )

with m2:
    render_html(
        """
        <div class="metric-card-gradient grad-2">
            <div style="font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; opacity: 0.9;">Precision</div>
            <div style="font-size: 30px; font-weight: 900; margin: 4px 0; font-family: 'Outfit';">84.27%</div>
            <div style="font-size: 11px; opacity: 0.85;">Positive Value</div>
        </div>
        """
    )

with m3:
    render_html(
        """
        <div class="metric-card-gradient grad-3">
            <div style="font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; opacity: 0.9;">Recall</div>
            <div style="font-size: 30px; font-weight: 900; margin: 4px 0; font-family: 'Outfit';">96.15%</div>
            <div style="font-size: 11px; opacity: 0.85;">Sensitivity Rate</div>
        </div>
        """
    )

with m4:
    render_html(
        """
        <div class="metric-card-gradient grad-4">
            <div style="font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; opacity: 0.9;">F1 Score</div>
            <div style="font-size: 30px; font-weight: 900; margin: 4px 0; font-family: 'Outfit';">89.82%</div>
            <div style="font-size: 11px; opacity: 0.85;">Harmonic Mean</div>
        </div>
        """
    )

with m5:
    render_html(
        """
        <div class="metric-card-gradient grad-5">
            <div style="font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; opacity: 0.9;">ROC-AUC</div>
            <div style="font-size: 30px; font-weight: 900; margin: 4px 0; font-family: 'Outfit';">94.82%</div>
            <div style="font-size: 11px; opacity: 0.85;">Discrimination Area</div>
        </div>
        """
    )

st.divider()

# ============================================================
# CONFUSION MATRIX
# ============================================================

render_html(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <span style="font-size: 24px;">📈</span>
        <h2 style="font-size: 22px; font-weight: 800; margin: 0; color: #0f172a;">Test Set Confusion Matrix</h2>
    </div>
    <p style="font-size: 14px; color: #64748b; margin-bottom: 15px;">
        Evaluation across all 624 hold-out test images demonstrating strong sensitivity on pneumonia cases.
    </p>
    """
)

cm = np.array([
    [164, 70],
    [15, 375]
])

fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["NORMAL", "PNEUMONIA"], fontsize=11, fontweight="bold")
ax.set_yticklabels(["NORMAL", "PNEUMONIA"], fontsize=11, fontweight="bold")

ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=8)
ax.set_ylabel("Actual Class", fontsize=12, fontweight="bold", labelpad=8)
ax.set_title("Test Confusion Matrix (N = 624)", fontsize=13, fontweight="bold", pad=12)

for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > 200 else "black"
        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=color
        )

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

col_cm1, col_cm2 = st.columns([1, 1])
with col_cm1:
    st.pyplot(fig)
plt.close(fig)

with col_cm2:
    render_html(
        """
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Key Takeaways:</div>
            <ul style="color: #475569; font-size: 14px; line-height: 1.7; padding-left: 18px; margin: 0;">
                <li><strong>High Recall (96.15%):</strong> Only 15 pneumonia cases were misclassified as normal out of 390.</li>
                <li><strong>Robust Discrimination:</strong> High ROC-AUC (94.82%) on independent clinical test data.</li>
                <li><strong>Balanced Performance:</strong> Strong overall accuracy (86.38%) on diverse X-ray acquisitions.</li>
            </ul>
        </div>
        """
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

render_html(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span style="font-size: 24px;">📁</span>
        <h2 style="font-size: 22px; font-weight: 800; margin: 0; color: #0f172a;">Dataset Distribution</h2>
    </div>
    """
)

d1, d2, d3, d4 = st.columns(4)

with d1:
    render_html(
        """
        <div class="data-metric-card">
            <div style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;">Training Set</div>
            <div style="font-size: 26px; font-weight: 800; color: #0ea5e9; margin: 4px 0;">4,710</div>
            <div style="font-size: 11px; color: #94a3b8;">Augmented Images</div>
        </div>
        """
    )

with d2:
    render_html(
        """
        <div class="data-metric-card">
            <div style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;">Validation Set</div>
            <div style="font-size: 26px; font-weight: 800; color: #6366f1; margin: 4px 0;">522</div>
            <div style="font-size: 11px; color: #94a3b8;">Tuning & Checkpoints</div>
        </div>
        """
    )

with d3:
    render_html(
        """
        <div class="data-metric-card">
            <div style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;">Hold-out Test</div>
            <div style="font-size: 26px; font-weight: 800; color: #10b981; margin: 4px 0;">624</div>
            <div style="font-size: 11px; color: #94a3b8;">Independent Test</div>
        </div>
        """
    )

with d4:
    render_html(
        """
        <div class="data-metric-card">
            <div style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;">Total Images</div>
            <div style="font-size: 26px; font-weight: 800; color: #8b5cf6; margin: 4px 0;">5,856</div>
            <div style="font-size: 11px; color: #94a3b8;">Full Cohort</div>
        </div>
        """
    )


# ============================================================
# SYSTEM WORKFLOW (INTERACTIVE 5-STEP PIPELINE)
# ============================================================

st.divider()

render_html(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span style="font-size: 24px;">🔄</span>
        <h2 style="font-size: 22px; font-weight: 800; margin: 0; color: #0f172a;">End-to-End AI System Workflow</h2>
    </div>
    """
)

w1, w2, w3, w4, w5 = st.columns(5)

with w1:
    render_html(
        """
        <div class="workflow-card step-1">
            <div style="font-size: 28px; margin-bottom: 6px;">📤</div>
            <div style="font-weight: 800; font-size: 15px; color: #0f172a;">1. Upload</div>
            <p style="font-size: 12.5px; color: #64748b; margin: 6px 0 0 0;">Upload patient radiograph in JPEG/PNG.</p>
        </div>
        """
    )

with w2:
    render_html(
        """
        <div class="workflow-card step-2">
            <div style="font-size: 28px; margin-bottom: 6px;">⚙️</div>
            <div style="font-weight: 800; font-size: 15px; color: #0f172a;">2. Preprocess</div>
            <p style="font-size: 12.5px; color: #64748b; margin: 6px 0 0 0;">Resize to 224 × 224 & normalize channels.</p>
        </div>
        """
    )

with w3:
    render_html(
        """
        <div class="workflow-card step-3">
            <div style="font-size: 28px; margin-bottom: 6px;">🧠</div>
            <div style="font-weight: 800; font-size: 15px; color: #0f172a;">3. Neural Net</div>
            <p style="font-size: 12.5px; color: #64748b; margin: 6px 0 0 0;">EfficientNetB0 feature extraction.</p>
        </div>
        """
    )

with w4:
    render_html(
        """
        <div class="workflow-card step-4">
            <div style="font-size: 28px; margin-bottom: 6px;">🎯</div>
            <div style="font-weight: 800; font-size: 15px; color: #0f172a;">4. Classify</div>
            <p style="font-size: 12.5px; color: #64748b; margin: 6px 0 0 0;">NORMAL vs PNEUMONIA with confidence.</p>
        </div>
        """
    )

with w5:
    render_html(
        """
        <div class="workflow-card step-5">
            <div style="font-size: 28px; margin-bottom: 6px;">🩺</div>
            <div style="font-weight: 800; font-size: 15px; color: #0f172a;">5. Guidance</div>
            <p style="font-size: 12.5px; color: #64748b; margin: 6px 0 0 0;">Grad-CAM & supportive health guidance.</p>
        </div>
        """
    )


# ============================================================
# FINAL MEDICAL DISCLAIMER (Feature 10)
# ============================================================

render_medical_disclaimer()
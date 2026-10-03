"""
AI Health Guidance Module for Chest X-Ray AI Prototype

This module provides safe, general, educational health guidance based on
model classifications and optional reported symptoms.

CRITICAL MEDICAL SAFETY REQUIREMENTS:
- Never claim the AI has diagnosed the patient.
- Never state "You have pneumonia."
- Use phrasing: "The AI model classified this X-ray as [LABEL]."
- Clearly explain that an X-ray AI prediction is not a medical diagnosis.
- The patient should consult a qualified doctor/radiologist for proper evaluation.
- Do not prescribe medications or recommend antibiotics, steroids, inhalers, or dosage.
- Do not recommend stopping or changing existing medication.
- Do not make claims about recovery time for the individual patient.
- Provide only general supportive health information.
- Include emergency warning signs and advise immediate medical attention when appropriate.
"""

from typing import Dict, List, Optional


COMMON_SYMPTOMS = [
    "Cough",
    "Fever",
    "Shortness of breath",
    "Chest discomfort",
    "Fatigue",
    "None of these",
]

EMERGENCY_SYMPTOMS = [
    "Difficulty breathing or severe shortness of breath",
    "Chest pain, especially severe or worsening chest pain",
    "Blue, gray, or pale lips/face",
    "New confusion, unusual drowsiness, or difficulty staying awake",
    "Severe weakness or rapidly worsening symptoms",
    "Coughing up significant amounts of blood",
]

EMERGENCY_STATEMENT = (
    "If you are experiencing severe breathing difficulty, blue/gray lips or face, "
    "severe chest pain, new confusion, or another medical emergency, seek emergency "
    "medical care immediately."
)

GRADCAM_DISCLAIMER = (
    "Grad-CAM is an explanatory visualization showing image regions that "
    "influenced the model prediction. It is not a medical annotation and should "
    "not be interpreted as proof of disease location."
)

FINAL_MEDICAL_DISCLAIMER = (
    "This application is an educational and research prototype. AI-generated "
    "classifications are not medical diagnoses and may contain errors. Do not use "
    "this application to make medical decisions or to start, stop, or change "
    "treatment. Consult a qualified healthcare professional for interpretation of "
    "chest X-rays and appropriate medical care."
)

NORMAL_STRUCTURED_POINTS = [
    {"icon": "💧", "title": "Hydration", "text": "Stay hydrated throughout the day."},
    {"icon": "🛌", "title": "Rest & Sleep", "text": "Get adequate rest and sleep."},
    {"icon": "🚭", "title": "Clean Air", "text": "Avoid smoking and second-hand smoke."},
    {"icon": "🌫️", "title": "Environment", "text": "Avoid unnecessary exposure to dust, fumes, and air pollution."},
    {"icon": "🧼", "title": "Hand Hygiene", "text": "Maintain good hand hygiene."},
    {"icon": "🥗", "title": "Nutrition", "text": "Follow a balanced diet."},
    {"icon": "🏃", "title": "Physical Activity", "text": "Maintain appropriate physical activity when feeling well."},
    {"icon": "👨‍⚕️", "title": "Clinical Guidance", "text": "Follow advice from your healthcare professional."},
]

PNEUMONIA_STRUCTURED_POINTS = [
    {"icon": "🛌", "title": "Adequate Rest", "text": "Get adequate rest and sleep."},
    {"icon": "💧", "title": "Hydration", "text": "Drink enough fluids unless a healthcare professional has advised you to restrict fluids."},
    {"icon": "🍲", "title": "Nutritious Food", "text": "Eat nutritious foods as tolerated."},
    {"icon": "🚭", "title": "Avoid Smoke", "text": "Avoid smoking and second-hand smoke."},
    {"icon": "🚫", "title": "Avoid Alcohol", "text": "Avoid alcohol if you are unwell or taking medicines that interact with alcohol."},
    {"icon": "🪟", "title": "Ventilation", "text": "Keep your surroundings well ventilated."},
    {"icon": "📋", "title": "Follow Treatment Plan", "text": "Follow the treatment plan prescribed by your healthcare professional."},
    {"icon": "💊", "title": "Medicine Directions", "text": "Take prescribed medicines exactly as directed."},
    {"icon": "🛑", "title": "No Self-Medicating", "text": "Do not start antibiotics or other prescription medicines without medical advice."},
    {"icon": "🩺", "title": "Symptom Monitoring", "text": "Monitor symptoms and seek medical care if symptoms worsen."},
]

NORMAL_GUIDANCE_POINTS = [p["text"] for p in NORMAL_STRUCTURED_POINTS]
PNEUMONIA_SUPPORTIVE_POINTS = [p["text"] for p in PNEUMONIA_STRUCTURED_POINTS]


def get_health_guidance(
    prediction: str,
    selected_symptoms: Optional[List[str]] = None
) -> Dict:
    """
    Returns structured, safe educational health guidance based on model classification.
    Symptoms are strictly used for educational context and doctor discussion prep,
    never for altering classification or providing a diagnosis.
    """
    active_symptoms = []
    if selected_symptoms:
        active_symptoms = [s for s in selected_symptoms if s != "None of these"]

    if prediction.upper() == "PNEUMONIA":
        return {
            "classification_badge": "⚠️ AI Classification: PNEUMONIA",
            "classification_explanation": (
                "This is an AI-generated classification, not a medical diagnosis. "
                "Please consult a qualified healthcare professional for clinical evaluation."
            ),
            "guidance_title": "General Supportive Care Information",
            "guidance_points": PNEUMONIA_SUPPORTIVE_POINTS,
            "structured_points": PNEUMONIA_STRUCTURED_POINTS,
            "guidance_note": (
                "Important: These suggestions are general supportive measures and do not cure pneumonia. "
                "No specific medications, dosages, or personalized treatment plans are provided. "
                "Always adhere to the specific management plan given by your doctor."
            ),
            "medical_followup": (
                "Discuss this result and any symptoms with a qualified healthcare professional."
            ),
            "reported_symptoms": active_symptoms,
            "symptom_disclaimer": (
                "These symptoms are collected only for educational guidance and are not used to diagnose a condition."
            ),
        }
    else:
        return {
            "classification_badge": "AI Classification: NORMAL",
            "classification_explanation": (
                "The model did not identify features that led it to classify this image as pneumonia. "
                "However, this AI result does not rule out other medical conditions. "
                "If you have symptoms or health concerns, consult a healthcare professional."
            ),
            "guidance_title": "General Healthy-Lung Guidance",
            "guidance_points": NORMAL_GUIDANCE_POINTS,
            "structured_points": NORMAL_STRUCTURED_POINTS,
            "guidance_note": (
                "Important: These tips support general respiratory wellness and do not treat or cure any medical disease."
            ),
            "medical_followup": (
                "Discuss this result and any symptoms with a qualified healthcare professional."
            ),
            "reported_symptoms": active_symptoms,
            "symptom_disclaimer": (
                "These symptoms are collected only for educational guidance and are not used to diagnose a condition."
            ),
        }


def get_emergency_warning() -> Dict:
    """
    Returns emergency red flag warning signs and instructions.
    """
    return {
        "title": "🚨 When to Seek Medical Help",
        "intro": (
            "Urgent medical evaluation may be needed if the person has symptoms such as:"
        ),
        "symptoms": EMERGENCY_SYMPTOMS,
        "statement": EMERGENCY_STATEMENT,
    }


def get_medical_disclaimer() -> Dict:
    """
    Returns the application-level medical disclaimer.
    """
    return {
        "title": "⚠️ Medical Disclaimer",
        "text": FINAL_MEDICAL_DISCLAIMER,
    }


def get_gradcam_disclaimer() -> str:
    """
    Returns the Grad-CAM visualization disclaimer.
    """
    return GRADCAM_DISCLAIMER

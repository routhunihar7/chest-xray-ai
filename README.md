# 🩺 Chest X-Ray AI - Pneumonia Classification

A deep learning based research prototype for classifying chest X-ray images into **NORMAL** and **PNEUMONIA** classes using **EfficientNetB0 Transfer Learning** and **Grad-CAM Explainable AI**.

---

## 📌 Project Overview

This project uses deep learning to analyze chest X-ray images and classify them into two categories:

- NORMAL
- PNEUMONIA

The system also uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize image regions that contributed to the model's prediction.

The complete system is integrated into an interactive **Streamlit dashboard**.

> ⚠️ **Disclaimer:** This project is an educational/research prototype and is not a clinical diagnostic system. Predictions should not be used for medical decisions.

---

## 🎯 Objectives

- Develop a deep learning model for chest X-ray classification.
- Use transfer learning with EfficientNetB0.
- Apply image preprocessing and data augmentation.
- Evaluate the model using multiple classification metrics.
- Implement Grad-CAM for model explainability.
- Build an interactive Streamlit dashboard.
- Provide a complete end-to-end AI workflow.

---

## 🏗️ System Architecture

```text
                    Chest X-Ray Image
                           │
                           ▼
                   Image Preprocessing
                           │
                           ▼
                  Resize to 224 × 224
                           │
                           ▼
                  EfficientNetB0
                  Transfer Learning
                           │
                           ▼
                    AI Prediction
                    /           \
                   /             \
              NORMAL          PNEUMONIA
                   \             /
                    \           /
                     ▼         ▼
                       Grad-CAM
                           │
                           ▼
                 Visual Explanation
                           │
                           ▼
                  Streamlit Dashboard
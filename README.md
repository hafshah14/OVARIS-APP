# OVARIS - Ovarian Risk Assessment and Screening System

**OVARIS** is a professional early-screening system designed to evaluate the risk of Polycystic Ovary Syndrome (PCOS) in female patients. By combining Machine Learning (for clinical data processing) and Deep Learning (for ovarian ultrasound image classification) under a single visual dashboard, OVARIS offers a comprehensive diagnostic aid.

---

## Key Features

1. **Patient Clinical Screening Wizard**
   - A step-by-step questionnaire processing 38 user inputs (41 features total).
   - Real-time physiological indicators: Indeks Massa Tubuh (BMI), Waist-to-Hip Ratio (WHR), and FSH/LH ratio.
   - Comprehensive data validation preventing predictions on incomplete profiles.
2. **Explainable AI (XAI) via SHAP**
   - Extract and sort the Top 5 most influential factors behind the patient's risk probability score.
   - Matplotlib SHAP horizontal bar plots indicating negative/positive contributions to PCOS risk.
3. **Personalized Recommendations**
   - Automatically generates targeted lifestyle, dietary, and clinical advice depending on the dominant risk drivers analyzed by SHAP.
4. **Ovarian USG Scan Classification**
   - A CNN deep-learning model evaluating USG scans.
   - Computes risk percentage and confidence ratings.
   - Generates a **Grad-CAM attention heatmap** highlighting relevant oocytes and folicular clusters.

---

## Folder Structure

```text
OVARIS/
│
├── app.py                      # Main entrypoint, page routing, and style loading
│
├── assets/
│   ├── logo.png                # Brand logo image
│   ├── styles.css              # Custom CSS theme (Soft Pink, Lavender, & Purple)
│
├── models/
│   ├── model.h5                # Pre-trained CNN USG model (expects 64x64x3 input)
│   └── ovaris_final_full_model.joblib # Pre-trained Tabular RandomForest pipeline
│
├── pages/
│   ├── home.py                 # System introduction, purpose, and quick links
│   ├── patient_screening.py    # 9-step clinical data screening form and results
│   └── image_screening.py      # USG image upload panel and Grad-CAM outcomes
│
├── utils/
│   ├── preprocessing.py        # Tabular data parser and column mapper
│   ├── prediction.py           # Cached model loaders and inference calls
│   ├── shap_explainer.py       # SHAP explanation calculations and visualizations
│   ├── recommendation.py       # Automated patient advice engine
│   ├── visualization.py        # Plotly gauges and risk meters
│   └── image_utils.py          # USG array converters and Grad-CAM overlays
│
├── requirements.txt            # Dependency listings
└── README.md                   # Project documentation
```

---

## Installation & Setup

To run OVARIS locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd OVARIS
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## Models & Data Specifications

- **Tabular Model:** RandomForestClassifier trained on 41 clinical and biochemical features (including hormonal parameters like FSH, LH, AMH, TSH, and ultrasound follicle sizes). Column preprocessing and RandomOverSampler are integrated into the scikit-learn pipeline bundle.
- **USG Image Model:** Sequential CNN expecting inputs of size `(64, 64, 3)` predicting 2 classes: `Normal` and `PCOS` (evaluated from ovaries USG scans). Includes a runtime monkeypatch to ensure compatibility with Keras 3 versions.

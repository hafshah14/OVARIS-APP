import streamlit as st
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from utils.prediction import load_tabular_model_bundle

FEATURE_LABELS = {
    'age_yrs': 'Usia (Tahun)',
    'weight_kg': 'Berat Badan (kg)',
    'height_cm': 'Tinggi Badan (cm)',
    'bmi': 'Indeks Massa Tubuh (BMI)',
    'blood_group': 'Golongan Darah',
    'pulse_rate_bpm': 'Denyut Nadi (bpm)',
    'rr_breaths_min': 'Frekuensi Napas (/mnt)',
    'hb_g_dl': 'Hemoglobin (g/dL)',
    'cycle_r_i': 'Siklus Menstruasi Teratur',
    'cycle_length_days': 'Lama Siklus Menstruasi (hari)',
    'marraige_status_yrs': 'Lama Menikah (tahun)',
    'pregnant_y_n': 'Pernah Hamil',
    'no_of_aborptions': 'Jumlah Keguguran',
    'i_beta_hcg_miu_ml': 'Beta HCG I (mIU/mL)',
    'ii_beta_hcg_miu_ml': 'Beta HCG II (mIU/mL)',
    'fsh_miu_ml': 'Kadar FSH (mIU/mL)',
    'lh_miu_ml': 'Kadar LH (mIU/mL)',
    'fsh_lh': 'Rasio FSH/LH',
    'hip_inch': 'Lingkar Pinggul (inci)',
    'waist_inch': 'Lingkar Pinggang (inci)',
    'waist_hip_ratio': 'Waist-Hip Ratio (WHR)',
    'tsh_miu_l': 'Kadar TSH (mIU/L)',
    'amh_ng_ml': 'Kadar AMH (ng/mL)',
    'prl_ng_ml': 'Kadar Prolaktin (ng/mL)',
    'vit_d3_ng_ml': 'Vitamin D3 (ng/mL)',
    'prg_ng_ml': 'Kadar Progesteron (ng/mL)',
    'rbs_mg_dl': 'Gula Darah Sewaktu (mg/dL)',
    'weight_gain_y_n': 'Kenaikan Berat Badan',
    'hair_growth_y_n': 'Pertumbuhan Rambut Berlebih',
    'skin_darkening_y_n': 'Penggelapan Kulit',
    'hair_loss_y_n': 'Kerontokan Rambut',
    'pimples_y_n': 'Jerawat',
    'fast_food_y_n': 'Konsumsi Fast Food',
    'reg_exercise_y_n': 'Olahraga Rutin',
    'bp_systolic_mmhg': 'Tekanan Darah Sistolik (mmHg)',
    'bp_diastolic_mmhg': 'Tekanan Darah Diastolik (mmHg)',
    'follicle_no_l': 'Folikel Ovarium Kiri (Jumlah)',
    'follicle_no_r': 'Folikel Ovarium Kanan (Jumlah)',
    'avg_f_size_l_mm': 'Rerata Ukuran Folikel Kiri (mm)',
    'avg_f_size_r_mm': 'Rerata Ukuran Folikel Kanan (mm)',
    'endometrium_mm': 'Ketebalan Endometrium (mm)'
}

@st.cache_resource
def get_shap_explainer(_model):
    """
    Creates and caches a TreeExplainer for the Random Forest model.
    """
    return shap.TreeExplainer(_model)

def explain_patient_risk(raw_df):
    """
    Calculates SHAP values for the given patient inputs and extracts the top features.
    """
    bundle = load_tabular_model_bundle()
    pipeline = bundle["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    
    # Preprocess the data using pipeline preprocessor
    X_trans = preprocessor.transform(raw_df)
    
    # Run SHAP explainer
    explainer = get_shap_explainer(model)
    shap_values = explainer.shap_values(X_trans)
    
    # Extract values for Class 1 (PMOS)
    if isinstance(shap_values, list):
        patient_shap = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        patient_shap = shap_values[0, :, 1]
    else:
        patient_shap = shap_values[0] # Fallback
        
    features = list(raw_df.columns)
    
    # Combine feature names, values, SHAP values, and directions
    explanations = []
    for idx, feature_name in enumerate(features):
        val = raw_df.iloc[0][feature_name]
        shap_val = patient_shap[idx]
        
        val_str = str(val)
        if feature_name in ['pregnant_y_n', 'weight_gain_y_n', 'hair_growth_y_n', 
                            'skin_darkening_y_n', 'hair_loss_y_n', 'pimples_y_n', 
                            'fast_food_y_n', 'reg_exercise_y_n']:
            val_str = "Ya" if val == 1.0 else "Tidak"
        elif feature_name == 'cycle_r_i':
            val_str = "Teratur" if val == 2.0 else "Tidak Teratur"
        elif feature_name == 'blood_group':
            group_map = {11: "A", 13: "B", 17: "AB", 15: "O"}
            val_str = group_map.get(int(val), "Lainnya")
            
        explanations.append({
            'feature': feature_name,
            'label': FEATURE_LABELS.get(feature_name, feature_name),
            'value': val,
            'value_str': val_str,
            'shap': shap_val,
            'direction': 'Meningkatkan Risiko' if shap_val > 0 else 'Menurunkan Risiko'
        })
        
    # Sort by absolute SHAP value
    explanations_sorted = sorted(explanations, key=lambda x: abs(x['shap']), reverse=True)
    
    return explanations_sorted, explainer, patient_shap, X_trans

def generate_shap_bar_plot(sorted_explanations, max_display=7):
    """
    Creates a customized horizontal bar plot of the top contributing features.
    """
    top_exps = sorted_explanations[:max_display][::-1]
    
    labels = [x['label'] for x in top_exps]
    values = [x['shap'] for x in top_exps]
    
    # Soft rose and gray colors
    colors = ['#F5AFAF' if v > 0 else '#E5E7EB' for v in values]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor='none', height=0.6)
    
    # Add vertical line at zero
    ax.axvline(0, color='#6B7280', linewidth=0.8, linestyle='--')
    
    # Style plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.tick_params(colors='#3F3F46', labelsize=9)
    ax.set_title("Kontribusi Fitur Terhadap Risiko PMOS", fontsize=11, fontweight='bold', color='#3F3F46', pad=15)
    ax.set_xlabel("Nilai Kontribusi (SHAP)", fontsize=9, color='#6B7280')
    
    # Add text on bars
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"+{width:.3f}", 
                    va='center', ha='left', fontsize=8, color='#3F3F46', fontweight='semibold')
        else:
            ax.text(width - 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                    va='center', ha='right', fontsize=8, color='#3F3F46')
            
    plt.tight_layout()
    return fig

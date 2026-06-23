import pandas as pd

# Mappings for categorical variables
BLOOD_GROUP_MAPPING = {
    "A": 11,   # A+ is 11
    "B": 13,   # B+ is 13
    "AB": 17,  # AB+ is 17
    "O": 15    # O+ is 15
}

YES_NO_MAPPING = {
    "Ya": 1,
    "Tidak": 0
}

# Regular = 2, Irregular = 4
CYCLE_MAPPING = {
    "Ya": 2,      # Regular -> 2
    "Tidak": 4    # Irregular -> 4
}

FEATURE_ORDER = [
    'age_yrs', 'weight_kg', 'height_cm', 'bmi', 'blood_group',
    'pulse_rate_bpm', 'rr_breaths_min', 'hb_g_dl', 'cycle_r_i',
    'cycle_length_days', 'marraige_status_yrs', 'pregnant_y_n',
    'no_of_aborptions', 'i_beta_hcg_miu_ml', 'ii_beta_hcg_miu_ml',
    'fsh_miu_ml', 'lh_miu_ml', 'fsh_lh', 'hip_inch', 'waist_inch',
    'waist_hip_ratio', 'tsh_miu_l', 'amh_ng_ml', 'prl_ng_ml',
    'vit_d3_ng_ml', 'prg_ng_ml', 'rbs_mg_dl', 'weight_gain_y_n',
    'hair_growth_y_n', 'skin_darkening_y_n', 'hair_loss_y_n',
    'pimples_y_n', 'fast_food_y_n', 'reg_exercise_y_n',
    'bp_systolic_mmhg', 'bp_diastolic_mmhg', 'follicle_no_l',
    'follicle_no_r', 'avg_f_size_l_mm', 'avg_f_size_r_mm',
    'endometrium_mm'
]

def preprocess_tabular_input(raw_inputs):
    """
    Transforms user interface raw inputs into a processed DataFrame matching
    the exact columns and data types expected by the machine learning pipeline.
    """
    processed = {}
    
    # Simple direct values
    processed['age_yrs'] = float(raw_inputs['usia'])
    processed['weight_kg'] = float(raw_inputs['berat'])
    processed['height_cm'] = float(raw_inputs['tinggi'])
    
    # Body calculations
    processed['bmi'] = float(raw_inputs['bmi'])
    processed['waist_hip_ratio'] = float(raw_inputs['whr'])
    processed['waist_inch'] = float(raw_inputs['lingkar_pinggang'])
    processed['hip_inch'] = float(raw_inputs['lingkar_pinggul'])
    
    # Categoricals mapping
    processed['blood_group'] = float(BLOOD_GROUP_MAPPING.get(raw_inputs['gol_darah'], 15.0))
    processed['cycle_r_i'] = float(CYCLE_MAPPING.get(raw_inputs['siklus'], 2.0))
    processed['pregnant_y_n'] = float(YES_NO_MAPPING.get(raw_inputs['pernah_hamil'], 0.0))
    
    # Regular inputs
    processed['cycle_length_days'] = float(raw_inputs['lama_siklus'])
    processed['marraige_status_yrs'] = float(raw_inputs['lama_menikah'])
    processed['no_of_aborptions'] = float(raw_inputs['jumlah_keguguran'])
    
    # Symptoms
    processed['weight_gain_y_n'] = float(YES_NO_MAPPING.get(raw_inputs['kenaikan_bb'], 0.0))
    processed['hair_growth_y_n'] = float(YES_NO_MAPPING.get(raw_inputs['rambut_berlebih'], 0.0))
    processed['skin_darkening_y_n'] = float(YES_NO_MAPPING.get(raw_inputs['kulit_gelap'], 0.0))
    processed['hair_loss_y_n'] = float(YES_NO_MAPPING.get(raw_inputs['rambut_rontok'], 0.0))
    processed['pimples_y_n'] = float(YES_NO_MAPPING.get(raw_inputs['jerawat'], 0.0))
    
    # Lifestyle
    processed['fast_food_y_n'] = float(YES_NO_MAPPING.get(raw_inputs['fast_food'], 0.0))
    processed['reg_exercise_y_n'] = float(YES_NO_MAPPING.get(raw_inputs['olahraga'], 0.0))
    
    # Vitals
    processed['pulse_rate_bpm'] = float(raw_inputs['denyut_nadi'])
    processed['rr_breaths_min'] = float(raw_inputs['frekuensi_napas'])
    processed['hb_g_dl'] = float(raw_inputs['hemoglobin'])
    processed['bp_systolic_mmhg'] = float(raw_inputs['sistolik'])
    processed['bp_diastolic_mmhg'] = float(raw_inputs['diastolik'])
    
    # Lab results
    processed['i_beta_hcg_miu_ml'] = float(raw_inputs['beta_hcg_1'])
    processed['ii_beta_hcg_miu_ml'] = float(raw_inputs['beta_hcg_2'])
    processed['fsh_miu_ml'] = float(raw_inputs['fsh'])
    processed['lh_miu_ml'] = float(raw_inputs['lh'])
    processed['fsh_lh'] = float(raw_inputs['fsh_lh'])
    processed['tsh_miu_l'] = float(raw_inputs['tsh'])
    processed['amh_ng_ml'] = float(raw_inputs['amh'])
    processed['prl_ng_ml'] = float(raw_inputs['prolaktin'])
    processed['vit_d3_ng_ml'] = float(raw_inputs['vitamin_d3'])
    processed['prg_ng_ml'] = float(raw_inputs['progesteron'])
    processed['rbs_mg_dl'] = float(raw_inputs['gula_darah'])
    
    # USG results
    processed['follicle_no_l'] = float(raw_inputs['folikel_kiri'])
    processed['follicle_no_r'] = float(raw_inputs['folikel_kanan'])
    processed['avg_f_size_l_mm'] = float(raw_inputs['ukuran_kiri'])
    processed['avg_f_size_r_mm'] = float(raw_inputs['ukuran_kanan'])
    processed['endometrium_mm'] = float(raw_inputs['endometrium'])
    
    # Make sure DataFrame column order matches FEATURE_ORDER exactly
    df = pd.DataFrame([processed])
    return df[FEATURE_ORDER]

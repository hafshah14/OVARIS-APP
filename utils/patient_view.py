import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_tabular_input
from utils.prediction import predict_tabular_pmos
from utils.shap_explainer import explain_patient_risk, FEATURE_LABELS
from utils.recommendation import generate_recommendations
from utils.visualization import generate_gauge_chart, generate_probability_meter

TOTAL_STEPS = 9

def get_friendly_factor_desc(feature_name, value):
    """
    Translates technical feature outputs into clean, patient-friendly medical terms.
    """
    if feature_name == 'amh_ng_ml':
        return "Kadar AMH Tinggi" if value > 3.0 else "Kadar AMH Normal"
    elif feature_name == 'bmi':
        return "BMI Tinggi (Overweight/Obesitas)" if value > 25.0 else "Indeks Massa Tubuh (BMI) Normal"
    elif feature_name == 'cycle_r_i':
        return "Siklus Menstruasi Tidak Teratur" if value == 4.0 else "Siklus Menstruasi Teratur"
    elif feature_name == 'lh_miu_ml':
        return "Kadar Hormon LH Tinggi" if value > 8.0 else "Kadar Hormon LH Normal"
    elif feature_name == 'fsh_lh':
        return "Rasio FSH/LH Tinggi" if value > 2.0 else "Rasio FSH/LH Normal"
    elif feature_name == 'follicle_no_l':
        return "Jumlah Folikel Ovarium Kiri Tinggi" if value > 10 else "Jumlah Folikel Kiri Normal"
    elif feature_name == 'follicle_no_r':
        return "Jumlah Folikel Ovarium Kanan Tinggi" if value > 10 else "Jumlah Folikel Kanan Normal"
    elif feature_name == 'fast_food_y_n':
        return "Konsumsi Makanan Cepat Saji Tinggi" if value == 1.0 else "Pola Makan Cepat Saji Rendah"
    elif feature_name == 'reg_exercise_y_n':
        return "Kurang Berolahraga Rutin" if value == 0.0 else "Aktivitas Olahraga Cukup"
    elif feature_name == 'weight_gain_y_n':
        return "Kenaikan Berat Badan Signifikan" if value == 1.0 else "Berat Badan Stabil"
    elif feature_name == 'hair_growth_y_n':
        return "Pertumbuhan Rambut Berlebih (Hirsutisme)" if value == 1.0 else "Tanpa Gejala Hirsutisme"
    elif feature_name == 'skin_darkening_y_n':
        return "Penggelapan Kulit (Acanthosis Nigricans)" if value == 1.0 else "Warna Kulit Normal"
    elif feature_name == 'pimples_y_n':
        return "Gejala Jerawat Hormonal Membandel" if value == 1.0 else "Tanpa Masalah Jerawat"
    elif feature_name == 'hair_loss_y_n':
        return "Kerontokan Rambut (Alopecia)" if value == 1.0 else "Kondisi Rambut Normal"
    elif feature_name == 'endometrium_mm':
        return "Ketebalan Endometrium Tinggi" if value > 12.0 else "Ketebalan Endometrium Normal"
    elif feature_name == 'age_yrs':
        return "Faktor Usia Pasien"
    else:
        return FEATURE_LABELS.get(feature_name, feature_name)

def initialize_session_state():
    """
    Ensures all questionnaire input fields are initialized in Streamlit session state.
    """
    if "step" not in st.session_state:
        st.session_state.step = 1
        
    keys_defaults = {
        'nama': '',
        'usia': None,
        'gol_darah': '',
        'berat': None,
        'tinggi': None,
        'lingkar_pinggang': None,
        'lingkar_pinggul': None,
        'bmi': 0.0,
        'whr': 0.0,
        'siklus': '',
        'lama_siklus': None,
        'lama_menikah': None,
        'pernah_hamil': '',
        'jumlah_keguguran': None,
        'kenaikan_bb': '',
        'rambut_berlebih': '',
        'kulit_gelap': '',
        'rambut_rontok': '',
        'jerawat': '',
        'fast_food': '',
        'olahraga': '',
        'denyut_nadi': None,
        'frekuensi_napas': None,
        'hemoglobin': None,
        'sistolik': None,
        'diastolik': None,
        'beta_hcg_1': None,
        'beta_hcg_2': None,
        'fsh': None,
        'lh': None,
        'fsh_lh': 0.0,
        'tsh': None,
        'amh': None,
        'prolaktin': None,
        'vitamin_d3': None,
        'progesteron': None,
        'gula_darah': None,
        'folikel_kiri': None,
        'folikel_kanan': None,
        'ukuran_kiri': None,
        'ukuran_kanan': None,
        'endometrium': None,
        'prediction_done': False,
        'prediction_prob': 0.0,
        'prediction_label': 0,
        'top_shap_features': []
    }
    
    for k, v in keys_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_completeness():
    """
    Calculates the count and percentage of completed required features.
    """
    required_keys = [
        'usia', 'gol_darah', 'berat', 'tinggi', 'lingkar_pinggang', 'lingkar_pinggul',
        'siklus', 'lama_siklus', 'lama_menikah', 'pernah_hamil', 'jumlah_keguguran',
        'kenaikan_bb', 'rambut_berlebih', 'kulit_gelap', 'rambut_rontok', 'jerawat',
        'fast_food', 'olahraga', 'denyut_nadi', 'frekuensi_napas', 'hemoglobin',
        'sistolik', 'diastolik', 'beta_hcg_1', 'beta_hcg_2', 'fsh', 'lh', 'tsh',
        'amh', 'prolaktin', 'vitamin_d3', 'progesteron', 'gula_darah',
        'folikel_kiri', 'folikel_kanan', 'ukuran_kiri', 'ukuran_kanan', 'endometrium'
    ]
    
    filled_count = 0
    empty_keys = []
    
    for key in required_keys:
        val = st.session_state[key]
        if val is not None and val != "":
            filled_count += 1
        else:
            empty_keys.append(key)
            
    percentage = (filled_count / len(required_keys)) * 100
    return filled_count, len(required_keys), percentage, empty_keys

def render_step_indicator(current_step):
    """
    Renders a clean HTML step indicator at the top of the wizard.
    """
    steps_html = '<div class="step-container">'
    for s in range(1, TOTAL_STEPS + 1):
        status_class = "step-bubble"
        if s == current_step:
            status_class = "step-bubble active"
        elif s < current_step:
            status_class = "step-bubble completed"
            
        steps_html += f'<div class="{status_class}">{s}</div>'
    steps_html += '</div>'
    
    st.write(steps_html, unsafe_allow_html=True)

def show_patient_screening():
    """
    Renders the Patient Clinical Questionnaire page.
    """
    initialize_session_state()
    
    st.markdown('<h2 style="color: #3F3F46; margin-bottom: 5px; font-family: Outfit;">Skrining Data Pasien</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #6B7280; font-size: 0.9rem;'>Isi seluruh formulir kesehatan dasar di bawah untuk menaksir risiko indikasi PMOS.</p>", unsafe_allow_html=True)
    
    # Render progress bubbles
    render_step_indicator(st.session_state.step)
    
    # Calculate global data completeness
    filled, total, comp_pct, empty_keys = get_completeness()
    
    step = st.session_state.step
    
    # Clean white container block
    step_container = st.container(border=True)
    
    with step_container:
        # ==========================
        # STEP 1: Data Pribadi
        # ==========================
        if step == 1:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Data Pribadi</h4>", unsafe_allow_html=True)
            
            st.session_state.nama = st.text_input(
                "Nama Lengkap Pasien (Opsional)",
                value=st.session_state.nama
            )
            
            st.session_state.usia = st.number_input(
                "Usia Pasien (Tahun) *",
                min_value=10,
                max_value=60,
                step=1,
                value=st.session_state.usia,
                placeholder="Masukkan usia"
            )
            
            gol_darah_options = ["", "A", "B", "AB", "O"]
            default_gol_idx = gol_darah_options.index(st.session_state.gol_darah) if st.session_state.gol_darah in gol_darah_options else 0
            
            st.session_state.gol_darah = st.selectbox(
                "Golongan Darah *",
                gol_darah_options,
                index=default_gol_idx
            )
            
            step_complete = (
                st.session_state.usia is not None 
                and st.session_state.gol_darah != ""
            )

        # ==========================
        # STEP 2: Ukuran Tubuh
        # ==========================
        elif step == 2:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Ukuran Tubuh</h4>", unsafe_allow_html=True)
            
            col_bb, col_tb = st.columns(2)
            with col_bb:
                st.session_state.berat = st.number_input(
                    "Berat Badan (kg) *",
                    min_value=25.0,
                    max_value=200.0,
                    step=0.1,
                    value=st.session_state.berat,
                    placeholder="Contoh: 54.5"
                )
            with col_tb:
                st.session_state.tinggi = st.number_input(
                    "Tinggi Badan (cm) *",
                    min_value=100.0,
                    max_value=220.0,
                    step=0.5,
                    value=st.session_state.tinggi,
                    placeholder="Contoh: 156.0"
                )
                
            col_pinggang, col_pinggul = st.columns(2)
            with col_pinggang:
                st.session_state.lingkar_pinggang = st.number_input(
                    "Lingkar Pinggang (inci) *",
                    min_value=15.0,
                    max_value=60.0,
                    step=0.1,
                    value=st.session_state.lingkar_pinggang,
                    placeholder="Contoh: 28.0"
                )
            with col_pinggul:
                st.session_state.lingkar_pinggul = st.number_input(
                    "Lingkar Pinggul (inci) *",
                    min_value=15.0,
                    max_value=60.0,
                    step=0.1,
                    value=st.session_state.lingkar_pinggul,
                    placeholder="Contoh: 35.0"
                )
                
            # Real-time calculations
            bmi_val = 0.0
            whr_val = 0.0
            
            if st.session_state.berat and st.session_state.tinggi:
                bmi_val = st.session_state.berat / ((st.session_state.tinggi / 100.0) ** 2)
                st.session_state.bmi = bmi_val
                
            if st.session_state.lingkar_pinggang and st.session_state.lingkar_pinggul:
                whr_val = st.session_state.lingkar_pinggang / st.session_state.lingkar_pinggul
                st.session_state.whr = whr_val
                
            # Metrics display in real time using clean gray layouts
            m_col1, m_col2 = st.columns(2)
            if bmi_val > 0:
                with m_col1:
                    st.markdown(
                        f"""
                        <div class="metric-container-white">
                            <div class="metric-label-gray">Indeks Massa Tubuh (BMI)</div>
                            <div class="metric-value-dark">{bmi_val:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            if whr_val > 0:
                with m_col2:
                    st.markdown(
                        f"""
                        <div class="metric-container-white">
                            <div class="metric-label-gray">Waist-Hip Ratio (WHR)</div>
                            <div class="metric-value-dark">{whr_val:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            step_complete = (
                st.session_state.berat is not None 
                and st.session_state.tinggi is not None 
                and st.session_state.lingkar_pinggang is not None 
                and st.session_state.lingkar_pinggul is not None
            )

        # ==========================
        # STEP 3: Riwayat Menstruasi & Kehamilan
        # ==========================
        elif step == 3:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Riwayat Menstruasi dan Kehamilan</h4>", unsafe_allow_html=True)
            
            yes_no_options = ["", "Ya", "Tidak"]
            
            st.session_state.siklus = st.selectbox(
                "Apakah siklus menstruasi teratur? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.siklus) if st.session_state.siklus in yes_no_options else 0
            )
            
            st.session_state.lama_siklus = st.number_input(
                "Rata-rata lama siklus menstruasi (hari) *",
                min_value=1,
                max_value=120,
                step=1,
                value=st.session_state.lama_siklus,
                placeholder="Jumlah hari siklus"
            )
            
            st.session_state.lama_menikah = st.number_input(
                "Lama Menikah (Tahun) *",
                min_value=0.0,
                max_value=40.0,
                step=0.5,
                value=st.session_state.lama_menikah,
                placeholder="Isi 0 jika belum menikah"
            )
            
            st.session_state.pernah_hamil = st.selectbox(
                "Apakah pasien pernah hamil? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.pernah_hamil) if st.session_state.pernah_hamil in yes_no_options else 0
            )
            
            st.session_state.jumlah_keguguran = st.number_input(
                "Jumlah Keguguran *",
                min_value=0,
                max_value=20,
                step=1,
                value=st.session_state.jumlah_keguguran,
                placeholder="Isi 0 jika tidak pernah mengalami keguguran"
            )
            
            step_complete = (
                st.session_state.siklus != ""
                and st.session_state.lama_siklus is not None
                and st.session_state.lama_menikah is not None
                and st.session_state.pernah_hamil != ""
                and st.session_state.jumlah_keguguran is not None
            )

        # ==========================
        # STEP 4: Gejala
        # ==========================
        elif step == 4:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Gejala Klinis</h4>", unsafe_allow_html=True)
            
            yes_no_options = ["", "Ya", "Tidak"]
            
            st.session_state.kenaikan_bb = st.selectbox(
                "Mengalami Kenaikan Berat Badan? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.kenaikan_bb) if st.session_state.kenaikan_bb in yes_no_options else 0
            )
            
            st.session_state.rambut_berlebih = st.selectbox(
                "Pertumbuhan Rambut Berlebih? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.rambut_berlebih) if st.session_state.rambut_berlebih in yes_no_options else 0
            )
            
            st.session_state.kulit_gelap = st.selectbox(
                "Penggelapan Kulit? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.kulit_gelap) if st.session_state.kulit_gelap in yes_no_options else 0
            )
            
            st.session_state.rambut_rontok = st.selectbox(
                "Kerontokan Rambut? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.rambut_rontok) if st.session_state.rambut_rontok in yes_no_options else 0
            )
            
            st.session_state.jerawat = st.selectbox(
                "Jerawat? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.jerawat) if st.session_state.jerawat in yes_no_options else 0
            )
            
            step_complete = (
                st.session_state.kenaikan_bb != ""
                and st.session_state.rambut_berlebih != ""
                and st.session_state.kulit_gelap != ""
                and st.session_state.rambut_rontok != ""
                and st.session_state.jerawat != ""
            )

        # ==========================
        # STEP 5: Gaya Hidup
        # ==========================
        elif step == 5:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Gaya Hidup</h4>", unsafe_allow_html=True)
            
            yes_no_options = ["", "Ya", "Tidak"]
            
            st.session_state.fast_food = st.selectbox(
                "Konsumsi Makanan Cepat Saji? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.fast_food) if st.session_state.fast_food in yes_no_options else 0
            )
            
            st.session_state.olahraga = st.selectbox(
                "Olahraga Rutin? *",
                yes_no_options,
                index=yes_no_options.index(st.session_state.olahraga) if st.session_state.olahraga in yes_no_options else 0
            )
            
            step_complete = (
                st.session_state.fast_food != ""
                and st.session_state.olahraga != ""
            )

        # ==========================
        # STEP 6: Pemeriksaan Dasar
        # ==========================
        elif step == 6:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Pemeriksaan Dasar</h4>", unsafe_allow_html=True)
            
            col_nadi, col_napas, col_hb = st.columns(3)
            with col_nadi:
                st.session_state.denyut_nadi = st.number_input(
                    "Denyut Nadi (bpm) *",
                    min_value=30,
                    max_value=200,
                    step=1,
                    value=st.session_state.denyut_nadi,
                    placeholder="Denyut nadi"
                )
            with col_napas:
                st.session_state.frekuensi_napas = st.number_input(
                    "Frekuensi Napas (/menit) *",
                    min_value=8,
                    max_value=60,
                    step=1,
                    value=st.session_state.frekuensi_napas,
                    placeholder="Frekuensi napas"
                )
            with col_hb:
                st.session_state.hemoglobin = st.number_input(
                    "Hemoglobin (g/dL) *",
                    min_value=4.0,
                    max_value=25.0,
                    step=0.1,
                    value=st.session_state.hemoglobin,
                    placeholder="Kadar Hb"
                )
                
            col_sistolik, col_diastolik = st.columns(2)
            with col_sistolik:
                st.session_state.sistolik = st.number_input(
                    "Tekanan Darah Sistolik (mmHg) *",
                    min_value=50,
                    max_value=250,
                    step=1,
                    value=st.session_state.sistolik,
                    placeholder="Sistolik"
                )
            with col_diastolik:
                st.session_state.diastolik = st.number_input(
                    "Tekanan Darah Diastolik (mmHg) *",
                    min_value=30,
                    max_value=150,
                    step=1,
                    value=st.session_state.diastolik,
                    placeholder="Diastolik"
                )
                
            step_complete = (
                st.session_state.denyut_nadi is not None
                and st.session_state.frekuensi_napas is not None
                and st.session_state.hemoglobin is not None
                and st.session_state.sistolik is not None
                and st.session_state.diastolik is not None
            )

        # ==========================
        # STEP 7: Hasil Laboratorium
        # ==========================
        elif step == 7:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Hasil Laboratorium</h4>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.beta_hcg_1 = st.number_input(
                    "Beta HCG I (mIU/mL) *",
                    min_value=0.0,
                    max_value=500.0,
                    step=0.01,
                    value=st.session_state.beta_hcg_1,
                    placeholder="Beta HCG I"
                )
            with c2:
                st.session_state.beta_hcg_2 = st.number_input(
                    "Beta HCG II (mIU/mL) *",
                    min_value=0.0,
                    max_value=500.0,
                    step=0.01,
                    value=st.session_state.beta_hcg_2,
                    placeholder="Beta HCG II"
                )
            with c3:
                st.session_state.tsh = st.number_input(
                    "TSH (mIU/L) *",
                    min_value=0.0,
                    max_value=50.0,
                    step=0.01,
                    value=st.session_state.tsh,
                    placeholder="TSH"
                )
                
            c4, c5, c6 = st.columns(3)
            with c4:
                st.session_state.fsh = st.number_input(
                    "FSH (mIU/mL) *",
                    min_value=0.0,
                    max_value=150.0,
                    step=0.01,
                    value=st.session_state.fsh,
                    placeholder="FSH"
                )
            with c5:
                st.session_state.lh = st.number_input(
                    "LH (mIU/mL) *",
                    min_value=0.0,
                    max_value=150.0,
                    step=0.01,
                    value=st.session_state.lh,
                    placeholder="LH"
                )
            with c6:
                st.session_state.amh = st.number_input(
                    "AMH (ng/mL) *",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    value=st.session_state.amh,
                    placeholder="AMH"
                )
                
            c7, c8, c9 = st.columns(3)
            with c7:
                st.session_state.prolaktin = st.number_input(
                    "Prolaktin (ng/mL) *",
                    min_value=0.0,
                    max_value=200.0,
                    step=0.01,
                    value=st.session_state.prolaktin,
                    placeholder="Prolaktin"
                )
            with c8:
                st.session_state.vitamin_d3 = st.number_input(
                    "Vitamin D3 (ng/mL) *",
                    min_value=0.0,
                    max_value=200.0,
                    step=0.01,
                    value=st.session_state.vitamin_d3,
                    placeholder="Vit D3"
                )
            with c9:
                st.session_state.progesteron = st.number_input(
                    "Progesteron (ng/mL) *",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    value=st.session_state.progesteron,
                    placeholder="Progesteron"
                )
                
            st.session_state.gula_darah = st.number_input(
                "Gula Darah Sewaktu (mg/dL) *",
                min_value=30.0,
                max_value=500.0,
                step=0.1,
                value=st.session_state.gula_darah,
                placeholder="Gula Darah"
            )
            
            # Real-time ratio
            ratio = 0.0
            if st.session_state.fsh and st.session_state.lh:
                ratio = st.session_state.fsh / st.session_state.lh
                st.session_state.fsh_lh = ratio
                
                st.markdown(
                    f"""
                    <div class="metric-container-white" style="max-width: 250px; margin-top: 15px;">
                        <div class="metric-label-gray">Rasio FSH / LH</div>
                        <div class="metric-value-dark">{ratio:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            step_complete = (
                st.session_state.beta_hcg_1 is not None
                and st.session_state.beta_hcg_2 is not None
                and st.session_state.fsh is not None
                and st.session_state.lh is not None
                and st.session_state.tsh is not None
                and st.session_state.amh is not None
                and st.session_state.prolaktin is not None
                and st.session_state.vitamin_d3 is not None
                and st.session_state.progesteron is not None
                and st.session_state.gula_darah is not None
            )

        # ==========================
        # STEP 8: Hasil USG Ovarium
        # ==========================
        elif step == 8:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Hasil USG</h4>", unsafe_allow_html=True)
            
            col_kiri, col_kanan = st.columns(2)
            with col_kiri:
                st.session_state.folikel_kiri = st.number_input(
                    "Jumlah Folikel Ovarium Kiri *",
                    min_value=0,
                    max_value=100,
                    step=1,
                    value=st.session_state.folikel_kiri,
                    placeholder="Jumlah kiri"
                )
                st.session_state.ukuran_kiri = st.number_input(
                    "Rata-rata Ukuran Folikel Kiri (mm) *",
                    min_value=0.0,
                    max_value=50.0,
                    step=0.1,
                    value=st.session_state.ukuran_kiri,
                    placeholder="Ukuran kiri"
                )
            with col_kanan:
                st.session_state.folikel_kanan = st.number_input(
                    "Jumlah Folikel Ovarium Kanan *",
                    min_value=0,
                    max_value=100,
                    step=1,
                    value=st.session_state.folikel_kanan,
                    placeholder="Jumlah kanan"
                )
                st.session_state.ukuran_kanan = st.number_input(
                    "Rata-rata Ukuran Folikel Kanan (mm) *",
                    min_value=0.0,
                    max_value=50.0,
                    step=0.1,
                    value=st.session_state.ukuran_kanan,
                    placeholder="Ukuran kanan"
                )
                
            st.session_state.endometrium = st.number_input(
                "Ketebalan Endometrium (mm) *",
                min_value=0.0,
                max_value=40.0,
                step=0.1,
                value=st.session_state.endometrium,
                placeholder="Ketebalan endometrium"
            )
            
            step_complete = (
                st.session_state.folikel_kiri is not None
                and st.session_state.folikel_kanan is not None
                and st.session_state.ukuran_kiri is not None
                and st.session_state.ukuran_kanan is not None
                and st.session_state.endometrium is not None
            )

        # ==========================
        # STEP 9: Ringkasan Data
        # ==========================
        elif step == 9:
            st.markdown("<h4 style='margin-top:0; color:#3F3F46;'>Ringkasan Data</h4>", unsafe_allow_html=True)
            
            c_pribadi, c_tubuh = st.columns(2)
            with c_pribadi:
                st.markdown(
                    f"""
                    <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:20px; height:100%; color:#3F3F46;">
                        <h4 style="margin-top:0; color:#3F3F46; font-size:1.05rem;">Profil Kesehatan Pasien</h4>
                        <b>Nama:</b> {st.session_state.nama if st.session_state.nama else '-'}<br>
                        <b>Usia:</b> {st.session_state.usia} tahun<br>
                        <b>Gol. Darah:</b> {st.session_state.gol_darah}<br>
                        <b>Siklus Menstruasi:</b> {st.session_state.siklus} (Teratur)<br>
                        <b>Lama Siklus:</b> {st.session_state.lama_siklus} hari<br>
                        <b>Lama Menikah:</b> {st.session_state.lama_menikah} tahun<br>
                        <b>Pernah Hamil:</b> {st.session_state.pernah_hamil} (Keguguran: {st.session_state.jumlah_keguguran})
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_tubuh:
                st.markdown(
                    f"""
                    <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:20px; height:100%; color:#3F3F46;">
                        <h4 style="margin-top:0; color:#3F3F46; font-size:1.05rem;">Ukuran Fisik & Gaya Hidup</h4>
                        <b>Berat Badan:</b> {st.session_state.berat} kg<br>
                        <b>Tinggi Badan:</b> {st.session_state.tinggi} cm<br>
                        <b>BMI:</b> {st.session_state.bmi:.2f}<br>
                        <b>WHR:</b> {st.session_state.whr:.3f}<br>
                        <b>Fast Food:</b> {st.session_state.fast_food}<br>
                        <b>Olahraga:</b> {st.session_state.olahraga}<br>
                        <b>Tekanan Darah:</b> {st.session_state.sistolik}/{st.session_state.diastolik} mmHg
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.write("")
            
            with st.expander("Tinjau Hasil Laboratorium & USG"):
                lab_data = {
                    "Pemeriksaan": [
                        "Beta HCG I", "Beta HCG II", "FSH", "LH", "Rasio FSH/LH", 
                        "TSH", "AMH", "Prolaktin", "Vitamin D3", "Progesteron", "Gula Darah Sewaktu (RBS)",
                        "Jumlah Folikel Kiri", "Jumlah Folikel Kanan", "Ukuran Folikel Kiri", "Ukuran Folikel Kanan", "Ketebalan Endometrium"
                    ],
                    "Nilai": [
                        f"{st.session_state.beta_hcg_1} mIU/mL", f"{st.session_state.beta_hcg_2} mIU/mL",
                        f"{st.session_state.fsh} mIU/mL", f"{st.session_state.lh} mIU/mL", f"{st.session_state.fsh_lh:.3f}",
                        f"{st.session_state.tsh} mIU/L", f"{st.session_state.amh} ng/mL", f"{st.session_state.prolaktin} ng/mL",
                        f"{st.session_state.vitamin_d3} ng/mL", f"{st.session_state.progesteron} ng/mL", f"{st.session_state.gula_darah} mg/dL",
                        f"{st.session_state.folikel_kiri} folikel", f"{st.session_state.folikel_kanan} folikel",
                        f"{st.session_state.ukuran_kiri} mm", f"{st.session_state.ukuran_kanan} mm", f"{st.session_state.endometrium} mm"
                    ]
                }
                st.table(pd.DataFrame(lab_data))
                
            # Completeness logic
            st.markdown("<h4 style='color: #3F3F46; font-size:1.1rem; margin-top:20px;'>Status Kelengkapan Data</h4>", unsafe_allow_html=True)
            st.write(f"Persentase pengisian data: **{comp_pct:.1f}%** ({filled} dari {total} data terisi)")
            st.progress(comp_pct / 100.0)
            
            if comp_pct < 100.0:
                st.markdown(
                    """
                    <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-left:4px solid #6B7280; padding:15px; border-radius:8px; margin-top:15px; font-size:0.9rem; color:#6B7280;">
                        Harap lengkapi seluruh data yang diperlukan sebelum melanjutkan.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                with st.expander("Lihat parameter yang kosong"):
                    missing_labels = [FEATURE_LABELS.get(key, key) for key in empty_keys]
                    st.write(", ".join(missing_labels))
                    
            step_complete = (comp_pct >= 100.0)

    # ==========================
    # NAVIGATION BUTTONS
    # ==========================
    col_prev, col_next = st.columns([1, 1])
    
    with col_prev:
        if step > 1:
            if st.button("Kembali", use_container_width=True):
                st.session_state.step -= 1
                st.session_state.prediction_done = False
                st.rerun()
                
    with col_next:
        if step < TOTAL_STEPS:
            if st.button("Lanjut", use_container_width=True, disabled=not step_complete if step in [1,2,3,4,5,6,7,8] else False):
                st.session_state.step += 1
                st.rerun()
        elif step == TOTAL_STEPS:
            # Custom styled prediction button (No red/purple, no icon)
            predict_btn = st.button("Analisis Risiko PMOS", type="primary", use_container_width=True, disabled=not step_complete)
            
            if predict_btn:
                with st.spinner("Menganalisis data pasien..."):
                    try:
                        raw_data = {
                            "usia": st.session_state.usia,
                            "berat": st.session_state.berat,
                            "tinggi": st.session_state.tinggi,
                            "bmi": st.session_state.bmi,
                            "gol_darah": st.session_state.gol_darah,
                            "denyut_nadi": st.session_state.denyut_nadi,
                            "frekuensi_napas": st.session_state.frekuensi_napas,
                            "hemoglobin": st.session_state.hemoglobin,
                            "siklus": st.session_state.siklus,
                            "lama_siklus": st.session_state.lama_siklus,
                            "lama_menikah": st.session_state.lama_menikah,
                            "pernah_hamil": st.session_state.pernah_hamil,
                            "jumlah_keguguran": st.session_state.jumlah_keguguran,
                            "beta_hcg_1": st.session_state.beta_hcg_1,
                            "beta_hcg_2": st.session_state.beta_hcg_2,
                            "fsh": st.session_state.fsh,
                            "lh": st.session_state.lh,
                            "fsh_lh": st.session_state.fsh_lh,
                            "lingkar_pinggul": st.session_state.lingkar_pinggul,
                            "lingkar_pinggang": st.session_state.lingkar_pinggang,
                            "whr": st.session_state.whr,
                            "tsh": st.session_state.tsh,
                            "amh": st.session_state.amh,
                            "prolaktin": st.session_state.prolaktin,
                            "vitamin_d3": st.session_state.vitamin_d3,
                            "progesteron": st.session_state.progesteron,
                            "gula_darah": st.session_state.gula_darah,
                            "kenaikan_bb": st.session_state.kenaikan_bb,
                            "rambut_berlebih": st.session_state.rambut_berlebih,
                            "kulit_gelap": st.session_state.kulit_gelap,
                            "rambut_rontok": st.session_state.rambut_rontok,
                            "jerawat": st.session_state.jerawat,
                            "fast_food": st.session_state.fast_food,
                            "olahraga": st.session_state.olahraga,
                            "sistolik": st.session_state.sistolik,
                            "diastolik": st.session_state.diastolik,
                            "folikel_kiri": st.session_state.folikel_kiri,
                            "folikel_kanan": st.session_state.folikel_kanan,
                            "ukuran_kiri": st.session_state.ukuran_kiri,
                            "ukuran_kanan": st.session_state.ukuran_kanan,
                            "endometrium": st.session_state.endometrium
                        }
                        
                        df_processed = preprocess_tabular_input(raw_data)
                        pred_label, pred_prob, threshold = predict_tabular_pmos(df_processed)
                        explanations, _, _, _ = explain_patient_risk(df_processed)
                        
                        st.session_state.prediction_prob = pred_prob
                        st.session_state.prediction_label = pred_label
                        st.session_state.top_shap_features = explanations
                        st.session_state.prediction_done = True
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat pemrosesan model: {e}")
                        
    # ==========================
    # DISPLAY SCREENING RESULTS
    # ==========================
    if st.session_state.prediction_done:
        st.markdown("<hr style='border:1px solid #E5E7EB; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #3F3F46; text-align: center; font-family:Outfit;'>Hasil Analisis Risiko PMOS</h3>", unsafe_allow_html=True)
        
        prob = st.session_state.prediction_prob
        
        res_col1, res_col2 = st.columns([1, 1], gap="large")
        
        with res_col1:
            st.markdown(
                """
                <div class="ovaris-card" style="text-align: center; display: flex; flex-direction: column; justify-content: center; height:100%;">
                    <h4 style="margin-top: 0; color:#3F3F46; font-size:1.1rem;">Probabilitas Risiko</h4>
                """,
                unsafe_allow_html=True
            )
            # Custom gauge chart
            gauge_fig = generate_gauge_chart(prob)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Simple probability meter
            prob_meter = generate_probability_meter(prob)
            st.plotly_chart(prob_meter, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(
                """
                <div class="ovaris-card" style="height:100%;">
                    <h4 style="margin-top: 0; color:#3F3F46; font-size:1.1rem;">Faktor Paling Berpengaruh</h4>
                    <p style="font-size:0.82rem; color:#6B7280; margin-bottom: 15px;">Berikut adalah parameter utama pasien yang memengaruhi skor probabilitas risiko model:</p>
                """,
                unsafe_allow_html=True
            )
            
            top_5 = st.session_state.top_shap_features[:5]
            for idx, item in enumerate(top_5, 1):
                # Rule-based patient friendly descriptions (no emoji/icons)
                friendly_desc = get_friendly_factor_desc(item['feature'], item['value'])
                impact_direction = "Meningkatkan risiko" if item['shap'] > 0 else "Menurunkan risiko"
                color_text = "#F5AFAF" if item['shap'] > 0 else "#6B7280"
                
                st.markdown(
                    f"""
                    <div style="padding: 10px 0; border-bottom: 1px solid #F3F4F6; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 500; color: #3F3F46; font-size:0.92rem;">{idx}. {friendly_desc}</span><br>
                            <span style="font-size: 0.8rem; color: #6B7280;">Nilai aktual: <b>{item['value_str']}</b></span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-weight: 500; color: {color_text}; font-size: 0.85rem;">{impact_direction}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        # ==========================
        # PERSONALIZED RECOMMENDATION
        # ==========================
        st.markdown("<h3 style='color: #3F3F46; margin-top: 30px; font-family: Outfit;'>Rekomendasi Personal</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#6B7280; font-size:0.9rem;'>Rekomendasi tindakan disusun secara khusus berdasarkan analisis parameter klinis utama Anda:</p>", unsafe_allow_html=True)
        
        recs = generate_recommendations(st.session_state.top_shap_features)
        
        for rec in recs:
            theme_class = "rec-card-warning" if rec['type'] == 'warning' else ("rec-card-info" if rec['type'] == 'info' else "rec-card-success")
            st.markdown(
                f"""
                <div class="{theme_class}">
                    <h4 style="margin: 0 0 6px 0; color: #3F3F46; font-size: 1rem; font-weight: 600;">
                        {rec['title']}
                    </h4>
                    <p style="font-size: 0.88rem; line-height: 1.6; margin: 0; color: #6B7280; text-align: justify;">
                        {rec['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Muted gray disclaimer
        st.write("")
        st.markdown(
            """
            <div class="disclaimer-card">
                <h4 style="color: #3F3F46; margin-bottom: 5px;">Disclaimer Hasil Skrining</h4>
                <p style="font-size: 0.85rem; line-height: 1.5; margin: 0; text-align: justify;">
                    Hasil analisis data klinis di atas hanya digunakan untuk skrining awal risiko PMOS. 
                    Segera diskusikan laporan evaluasi ini dengan dokter spesialis obstetri dan ginekologi (Sp.OG) untuk pemeriksaan 
                    fisik ginekologis serta tindakan diagnosis klinis menyeluruh.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

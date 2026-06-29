import streamlit as st
from PIL import Image
import numpy as np
from utils.image_utils import preprocess_image, generate_gradcam
from utils.prediction import predict_image_pmos, validate_usg_image_with_model
from utils.visualization import generate_gauge_chart

def show_image_screening():
    """
    Renders the USG Image Screening Page.
    Handles file upload, preview, CNN classification, and Grad-CAM heatmap visualization.
    No emojis or visual stickers are used.
    """
    st.markdown('<h2 style="color: #3F3F46; margin-bottom: 5px; font-family: Outfit;">Skrining Citra USG</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #6B7280; font-size: 0.9rem;'>Unggah citra ginekologis USG ovarium pasien untuk menganalisis risiko PMOS.</p>", unsafe_allow_html=True)
    st.write("")

    # Left: Uploader Card, Right: Preview Card
    col_upload, col_preview = st.columns([1, 1], gap="large")
    
    uploaded_file = None
    analyse_btn = False
    
    with col_upload:
        st.markdown(
            """
            <div class="ovaris-card" style="height: 100%;">
                <h4 style="margin-top:0; color:#3F3F46; font-size:1.1rem;">Unggah Pemindaian USG</h4>
                <p style="font-size:0.82rem; color:#6B7280; margin-bottom:15px; line-height:1.5;">
                    Pilih file citra hasil USG ovarium. Pastikan gambar memiliki kontras yang jelas dan fokus pada area ovarium.
                </p>
            """,
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Pilih file (Format: JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        st.write("")
        
        if uploaded_file is not None:
            # Check for new upload and reset state
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_uploaded_file") != file_key:
                st.session_state.last_uploaded_file = file_key
                st.session_state.img_prediction_done = False
                st.session_state.img_prediction_label = None
                st.session_state.img_prediction_prob = 0.0
                st.session_state.img_gradcam_result = None
                
                # Perform validation immediately
                with st.spinner("Memvalidasi format citra USG menggunakan AI..."):
                    try:
                        from utils.prediction import load_image_model
                        img_model = load_image_model()
                        input_shape = img_model.input_shape
                        target_size = (input_shape[1], input_shape[2])
                        img_tensor, orig_img = preprocess_image(uploaded_file, target_size=target_size)
                        is_valid, debug_info, val_message = validate_usg_image_with_model(uploaded_file, img_tensor)
                        st.session_state.is_valid_usg = is_valid
                        st.session_state.validation_message = val_message
                        
                        # Store additional shape debug info
                        debug_info["input_shape_model"] = str(input_shape)
                        debug_info["image_shape_preprocessed"] = f"{target_size[0]}x{target_size[1]}"
                        debug_info["tensor_shape_inference"] = str(img_tensor.shape)
                        
                        st.session_state.validation_debug_info = debug_info
                    except Exception as e:
                        st.session_state.is_valid_usg = True # Fallback if error
                        st.session_state.validation_message = f"Gagal validasi: {e}"
                        st.session_state.validation_debug_info = {"error": str(e)}
            
            # Display Validation Box in Left Card
            is_valid = st.session_state.get("is_valid_usg", True)
            if is_valid:
                st.markdown(
                    """
                    <div style="background-color: #F0FDF4; border-left: 4px solid #10B981; padding: 12px; border-radius: 6px; margin-bottom: 15px; text-align: left;">
                        <strong style="color: #15803D; font-size: 0.9rem;">Status Gambar</strong><br>
                        <span style="color: #166534; font-size: 0.82rem;">✅ Citra USG ovarium berhasil dikenali.</span><br>
                        <span style="color: #166534; font-size: 0.82rem;">Silakan lanjutkan dengan menekan tombol <b>Analisis Risiko PMOS</b>.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 12px; border-radius: 6px; margin-bottom: 15px; text-align: left;">
                        <strong style="color: #991B1B; font-size: 0.9rem;">Status Gambar</strong><br>
                        <span style="color: #991B1B; font-size: 0.82rem;">❌ Gambar tidak dikenali sebagai citra USG ovarium.</span><br>
                        <span style="color: #991B1B; font-size: 0.82rem;">Silakan unggah citra USG ovarium yang sesuai untuk melanjutkan proses skrining.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # The button is disabled if validation fails
            analyse_btn = st.button("Analisis Risiko PMOS", type="primary", use_container_width=True, disabled=not is_valid)
            
            # Print debug details to terminal console for development/audit logs
            debug_info = st.session_state.get("validation_debug_info")
            if debug_info and "error" not in debug_info:
                print("\n[OVARIS AUDIT LOG] - VALIDASI CITRA USG")
                print(f"  Input Shape Model : {debug_info.get('input_shape_model')}")
                print(f"  Preprocessed Size : {debug_info.get('image_shape_preprocessed')}")
                print(f"  Tensor Shape      : {debug_info.get('tensor_shape_inference')}")
                print(f"  1. Validasi File   : {debug_info.get('is_valid_file_msg')}")
                print(f"  2. Screening Warna : {debug_info.get('basic_screening_msg')}")
                print(f"  3. Coretan/Scribble: {debug_info.get('is_scribble_msg')}")
                print(f"  4. Objek non-USG   : {debug_info.get('is_non_usg_object_msg')}")
                print(f"  5. Jarak OOD       : min_dist={debug_info.get('min_dist'):.4f}, threshold={debug_info.get('threshold'):.4f}")
                print(f"  6. Raw Prediction  : {debug_info.get('raw_probs')}")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_preview:
        st.markdown(
            """
            <div class="ovaris-card" style="height: 100%; text-align: center;">
                <h4 style="margin-top:0; color:#3F3F46; font-size:1.1rem; text-align: left;">Pratinjau Citra USG</h4>
            """,
            unsafe_allow_html=True
        )
        if uploaded_file is not None:
            uploaded_file.seek(0)
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="height: 150px; display: flex; align-items: center; justify-content: center; border: 1px dashed #E5E7EB; border-radius: 8px; color: #6B7280; font-size: 0.85rem;">
                    Citra USG yang diunggah akan muncul di sini
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Process and run prediction on button click
    if uploaded_file is not None and analyse_btn:
        with st.spinner("Menganalisis gambar USG menggunakan model Deep Learning..."):
            try:
                # Load Keras CNN model to inspect target_size dynamically
                from utils.prediction import load_image_model
                img_model = load_image_model()
                
                # Fetch shape from model dynamically
                input_shape = img_model.input_shape
                target_size = (input_shape[1], input_shape[2])
                
                # Preprocess image with dynamic target_size
                img_tensor, orig_img = preprocess_image(uploaded_file, target_size=target_size)
                
                # Run prediction
                prediction_label, pmos_prob, raw_probs = predict_image_pmos(img_tensor)
                
                # Generate Grad-CAM heatmap dynamically finding last conv layer
                gradcam_img, heatmap = generate_gradcam(
                    model=img_model,
                    img_array=img_tensor,
                    original_img=orig_img,
                    last_conv_layer_name=None
                )
                
                # Save outputs to session state
                st.session_state.img_prediction_done = True
                st.session_state.img_prediction_label = prediction_label
                st.session_state.img_prediction_prob = float(pmos_prob)
                st.session_state.img_gradcam_result = gradcam_img
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses citra USG: {e}")

    # ==========================
    # DISPLAY IMAGE RESULTS
    # ==========================
    if uploaded_file is not None and st.session_state.get("img_prediction_done", False):
        st.markdown("<hr style='border:1px solid #E5E7EB; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #3F3F46; text-align: center; font-family: Outfit;'>Hasil Analisis Citra</h3>", unsafe_allow_html=True)
        
        p_prob = st.session_state.img_prediction_prob
        p_label = st.session_state.img_prediction_label
        
        res_c1, res_c2 = st.columns([1, 1], gap="large")
        
        with res_c1:
            st.markdown(
                """
                <div class="ovaris-card" style="text-align: center; height:100%;">
                    <h4 style="margin-top: 0; color:#3F3F46; font-size:1.1rem; text-align:left;">Skor Risiko & Prediksi</h4>
                """,
                unsafe_allow_html=True
            )
            
            # Show Plotly Gauge with new color palette
            gauge_fig = generate_gauge_chart(p_prob)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Prediction outcome card
            label_color = "#F5AFAF" if p_label == "PMOS" else "#6B7280"
            confidence_val = p_prob if p_label == "PMOS" else (1.0 - p_prob)
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 15px; margin-top: 10px;">
                    <span style="font-size:0.85rem; color:#6B7280;">Prediksi Model:</span><br>
                    <span style="font-size:1.8rem; font-weight:bold; color:{label_color};">{p_label}</span><br>
                    <span style="font-size:0.82rem; color:#6B7280;">Confidence Score: <b>{confidence_val*100:.2f}%</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with res_c2:
            st.markdown(
                """
                <div class="ovaris-card" style="text-align: center; height:100%;">
                    <h4 style="margin-top: 0; color:#3F3F46; font-size:1.1rem; text-align:left;">Visualisasi Grad-CAM</h4>
                    <p style="font-size:0.82rem; color:#6B7280; text-align:justify; margin-bottom:15px; line-height:1.5;">
                        Peta panas (heatmap) Grad-CAM menandakan area penampang pada USG ovarium yang menjadi fokus utama perhatian model saat mengevaluasi risiko PMOS.
                    </p>
                """,
                unsafe_allow_html=True
            )
            
            if st.session_state.img_gradcam_result is not None:
                st.image(st.session_state.img_gradcam_result, use_container_width=True)
            else:
                st.markdown(
                    """
                    <div style="height: 150px; display: flex; align-items: center; justify-content: center; border: 1px dashed #E5E7EB; border-radius: 8px; color: #6B7280; font-size: 0.85rem;">
                        Visualisasi Grad-CAM tidak dapat dimuat
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Recommendations & Advice Section
        st.write("")
        
        # Calculate Confidence Score: probability of predicted class
        confidence_val = p_prob if p_label == "PMOS" else (1.0 - p_prob)
        
        if p_label == "PMOS":
            recommendation_text = (
                "Hasil citra menunjukkan karakteristik ovarium yang perlu dievaluasi lebih lanjut. "
                "Disarankan untuk berkonsultasi dengan dokter spesialis obstetri dan ginekologi guna memperoleh "
                "pemeriksaan lanjutan serta interpretasi klinis yang sesuai."
            )
        else:
            recommendation_text = (
                "Tidak ditemukan karakteristik dominan yang mengarah ke PMOS pada citra yang dianalisis. "
                "Namun hasil ini hanya bersifat skrining awal dan tidak menggantikan pemeriksaan medis profesional."
            )
            
        st.markdown(
            f"""<div class="ovaris-card">
<h3 style="margin-top: 0; color: #3F3F46; font-size: 1.25rem; font-family: Outfit; margin-bottom: 20px;">Rekomendasi Analisis Citra</h3>
<div style="background: #FCF8F8; border-left: 4px solid #F5AFAF; padding: 18px; border-radius: 6px; margin-bottom: 25px;">
<p style="margin: 0; color: #6B7280; font-size: 0.9rem; line-height: 1.7; text-align: justify;">
{recommendation_text}
</p>
</div>
<div>
<h4 style="margin: 0 0 12px 0; color: #3F3F46; font-size: 0.98rem; font-family: Outfit; font-weight: 600;">Saran Umum</h4>
<ul style="margin: 0; padding-left: 20px; color: #6B7280; font-size: 0.88rem; line-height: 1.7; text-align: justify;">
<li style="margin-bottom: 8px;">Menjaga pola makan seimbang</li>
<li style="margin-bottom: 8px;">Menjaga berat badan ideal</li>
<li style="margin-bottom: 8px;">Melakukan aktivitas fisik secara rutin</li>
<li style="margin-bottom: 0;">Berkonsultasi dengan tenaga medis apabila mengalami gangguan siklus menstruasi atau keluhan reproduksi lainnya</li>
</ul>
</div>
</div>""",
            unsafe_allow_html=True
        )
        
        # Clinical Disclaimer
        st.write("")
        st.markdown(
            """
            <div class="disclaimer-card">
                <h4 style="color: #3F3F46; margin-bottom: 5px; font-family: Outfit;">Disclaimer Analisis Citra</h4>
                <p style="font-size: 0.85rem; line-height: 1.5; margin: 0; text-align: justify;">
                    Hasil analisis citra hanya digunakan untuk skrining awal dan tidak dapat menggantikan interpretasi dokter spesialis radiologi maupun dokter spesialis kandungan. 
                    Segala bentuk keputusan klinis atau terapi medis harus merujuk pada penilaian langsung dari praktisi obstetri dan ginekologi profesional.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

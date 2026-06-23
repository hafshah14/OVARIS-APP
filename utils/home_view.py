import streamlit as st
import base64
import os

def get_base64_image(image_path):
    """
    Reads a local image file and returns its base64 representation.
    Guarantees robust rendering inside HTML blocks across all server configurations.
    """
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    return ""

def show_home():
    """
    Renders the OVARIS Landing Page.
    Includes logo card, header information, Tentang OVARIS, Tujuan dan Manfaat, 
    Disclaimer, and Quick Access navigation tiles.
    No emojis or visual stickers are used.
    """
    # Load logo as base64
    logo_base64 = get_base64_image("assets/logo.png")
    if not logo_base64:
        logo_base64 = get_base64_image(os.path.join("app", "static", "assets", "logo.png"))
        
    logo_html = ""
    if logo_base64:
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />'
    else:
        # Fallback text if image not found
        logo_html = '<span style="font-family: Outfit; font-weight: 700; color: #F5AFAF; font-size: 1.5rem;">OVARIS</span>'
    
    # Hero Section Layout: Horizontal 2-Column Responsive Flexbox
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-logo-box">
                {logo_html}
            </div>
            <div class="hero-text-box">
                <h1>OVARIS</h1>
                <p class="tagline">AI-Based Early Screening System for PMOS</p>
                <p class="description">
                    Sistem skrining awal risiko PMOS (Polycystic Ovary Syndrome) berbasis kecerdasan buatan. 
                    Membantu mendeteksi risiko PMOS lebih awal melalui analisis data kesehatan dan gambar USG ovarium secara cepat dan mudah.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Grid: Grid Layout below Hero
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Tentang OVARIS Card
        st.markdown(
            """
            <div class="ovaris-card" style="height: 100%;">
                <h3 style="margin-top: 0; color: #3F3F46; font-size: 1.25rem; font-family: 'Outfit';">Tentang OVARIS</h3>
                <p style="font-size: 0.9rem; color: #6B7280; line-height: 1.7; text-align: justify; margin-bottom: 0;">
                    OVARIS (Ovarian Risk Assessment and Screening System) adalah aplikasi kesehatan digital yang 
                    dirancang untuk membantu melakukan skrining awal risiko PMOS. Aplikasi ini menggabungkan analisis data kesehatan pasien 
                    (seperti riwayat medis, hormon, gaya hidup) dan analisis gambar USG ovarium menggunakan teknologi kecerdasan buatan (AI) 
                    guna mendeteksi potensi risiko secara cepat dan praktis sebelum melakukan pemeriksaan medis lebih lanjut.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")  # spacing

        # Tujuan dan Manfaat Card
        st.markdown(
            """
            <div class="ovaris-card" style="height: 100%;">
                <h3 style="margin-top: 0; color: #3F3F46; font-size: 1.25rem; font-family: 'Outfit';">Tujuan dan Manfaat</h3>
                <ul style="font-size: 0.9rem; color: #6B7280; line-height: 1.7; margin: 0; padding-left: 20px; text-align: justify;">
                    <li style="margin-bottom: 8px;">Mempermudah masyarakat melakukan skrining risiko PMOS secara mandiri, cepat, dan praktis sebelum pemeriksaan medis mendalam.</li>
                    <li style="margin-bottom: 8px;">Membantu memberikan perkiraan tingkat risiko PMOS berdasarkan data kesehatan dan analisis gambar ovarium secara otomatis.</li>
                    <li style="margin-bottom: 0;">Memberikan saran awal serta rekomendasi gaya hidup sehat guna mendukung konsultasi yang lebih terarah dengan dokter spesialis kandungan.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        # Akses Skrining Cepat Section
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #3F3F46; font-size: 1.25rem; font-family: 'Outfit'; margin-bottom: 15px;">Akses Skrining Cepat</h3>
            """,
            unsafe_allow_html=True
        )
        
        q_col1, q_col2 = st.columns(2, gap="medium")
        
        with q_col1:
            st.markdown(
                """
                <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; height: 135px; background: #FCF8F8;">
                    <h4 style="margin: 0; color: #3F3F46; font-size: 0.95rem; font-family: 'Outfit';">Skrining Data</h4>
                    <p style="font-size: 0.78rem; color: #6B7280; margin-top: 6px; line-height: 1.4;">
                        Analisis data kesehatan pasien seperti riwayat medis, hormon, dan gaya hidup.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            if st.button("Mulai Skrining Data", key="btn_go_tabular", use_container_width=True):
                st.session_state.menu = "Skrining Data Pasien"
                st.rerun()
                
        with q_col2:
            st.markdown(
                """
                <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; height: 135px; background: #FCF8F8;">
                    <h4 style="margin: 0; color: #3F3F46; font-size: 0.95rem; font-family: 'Outfit';">Skrining Citra USG</h4>
                    <p style="font-size: 0.78rem; color: #6B7280; margin-top: 6px; line-height: 1.4;">
                        Analisis gambar hasil USG ovarium secara otomatis untuk mendeteksi potensi risiko.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            if st.button("Analisis USG", key="btn_go_image", use_container_width=True):
                st.session_state.menu = "Skrining Citra USG"
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)

        # Disclaimer Card
        st.markdown(
            """
            <div class="disclaimer-card" style="margin-bottom: 0;">
                <h4 style="color: #3F3F46; font-size: 1rem; margin-bottom: 8px; font-family: 'Outfit';">Disclaimer Medis</h4>
                <p style="font-size: 0.85rem; line-height: 1.6; margin: 0; text-align: justify; color: #6B7280;">
                    Hasil yang diberikan oleh sistem ini hanya digunakan untuk skrining awal dan tidak dapat digunakan sebagai pengganti diagnosis, konsultasi, maupun keputusan medis profesional. 
                    Seluruh interpretasi hasil skrining risiko PMOS harus dikonfirmasikan dan divalidasi oleh dokter spesialis kandungan (Sp.OG) yang berwenang.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

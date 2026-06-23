import streamlit as st
import os

# Set page configuration first before any other Streamlit calls
st.set_page_config(
    page_title="OVARIS - Ovarian Risk Assessment and Screening System",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Stylesheet
styles_path = os.path.join("assets", "styles.css")
if os.path.exists(styles_path):
    with open(styles_path, "r", encoding="utf-8") as f:
        css_style = f.read()
    st.markdown(f"<style>{css_style}</style>", unsafe_allow_html=True)

# Initialize navigation menu state
if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

# Minimalist Navigation Radio Options
menu_options = ["Beranda", "Skrining Data Pasien", "Skrining Citra USG"]
menu_idx = menu_options.index(st.session_state.menu) if st.session_state.menu in menu_options else 0

selected_menu = st.sidebar.radio(
    "NAVIGASI",
    menu_options,
    index=menu_idx
)

# Handle selection update and two-way sync
if selected_menu != st.session_state.menu:
    st.session_state.menu = selected_menu
    # If switching menus, reset wizard steps to keep memory clean
    if selected_menu != "Skrining Data Pasien" and "step" in st.session_state:
        st.session_state.step = 1
    st.rerun()

# Navigation routing
if st.session_state.menu == "Beranda":
    from utils.home_view import show_home
    show_home()
    
elif st.session_state.menu == "Skrining Data Pasien":
    from utils.patient_view import show_patient_screening
    show_patient_screening()
    
elif st.session_state.menu == "Skrining Citra USG":
    from utils.image_view import show_image_screening
    show_image_screening()
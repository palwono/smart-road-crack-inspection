import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np

from model import load_model, predict_crack_mask, overlay_mask
from utils import validate_image, compute_crack_metrics, determine_condition, log_inspection, load_inspection_log

# 1. KONFIGURASI HALAMAN (Sidebar tertutup otomatis ala Gemini)
st.set_page_config(
    page_title="AI Road Command", 
    page_icon="🚧", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# 2. INJEKSI CUSTOM CSS (UI ala Gemini & Perbaikan Judul)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700&family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Membatasi lebar konten agar terpusat (sentral) dan tidak melebar tak wajar */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1000px !important; 
    }
    
    /* Menyesuaikan ukuran font judul agar tidak terpotong */
    h1, h2, h3, h4, h5 {
        font-family: 'Chakra Petch', sans-serif !important;
        text-transform: uppercase;
        color: #F9A826 !important;
        margin-bottom: 0px !important;
    }

    /* --- SULAP SIDEBAR ALA GEMINI --- */
    /* Menyembunyikan bulatan radio button */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none; 
    }
    /* Mengubah gaya menu menjadi kotak (pill) yang rapi */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 12px 15px;
        border-radius: 8px;
        transition: all 0.2s;
        margin-bottom: 5px;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: rgba(255,255,255,0.1);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #F9A826;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p {
        color: #121212 !important;
        font-weight: 700;
    }

    /* Modifikasi Kotak Metrik */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #333;
        padding: 10px 15px;
        border-radius: 4px;
        border-left: 5px solid #F9A826;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

class SmartRoadApp:
    def __init__(self):
        self.model = load_model()
        if "processed_file" not in st.session_state:
            st.session_state.processed_file = None
            st.session_state.temp_metrics = None
            st.session_state.temp_condition = None
            st.session_state.temp_result_img = None
        
    def run(self):
        # Header Aplikasi Utama (Disesuaikan ukurannya)
        st.markdown("<h2 style='text-align: center; font-size: 2.2rem;'>🚧 SMART ROAD CRACK INSPECTION SYSTEM</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; font-size: 0.9rem;'>AI Semantic Segmentation & Infrastructure Analytics Engine</p>", unsafe_allow_html=True)
        st.divider()

        st.sidebar.markdown("### 🎛️ COMMAND MENU")
        menu = st.sidebar.radio("Navigasi Utama", ["DETECTION MODULE", "ANALYTICS DASHBOARD"], label_visibility="collapsed")
        
        if menu == "DETECTION MODULE":
            self.halaman_inspeksi()
        elif menu == "ANALYTICS DASHBOARD":
            self.halaman_dashboard()

    def halaman_inspeksi(self):
        # 1. AREA UPLOAD SENTRAL (Di Tengah Layar)
        _, col_center, _ = st.columns([1, 2, 1])
        
        with col_center:
            st.markdown("<h4 style='text-align: center;'>📤 IMAGE UPLOAD</h4>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Pilih foto jalan raya", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            if uploaded_file is None:
                st.info("💡 Klik ikon '>' di pojok kiri atas untuk membuka Sidebar Menu.")
        
        # 2. AREA HASIL (Akan muncul di bawah setelah gambar diunggah)
        if uploaded_file is not None:
            is_valid, error_msg = validate_image(uploaded_file)
            if not is_valid:
                st.error(error_msg)
                return

            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_input, col_result = st.columns([1, 1.5], gap="large")
            
            with col_input:
                st.image(image, use_container_width=True, caption="ORIGINAL INPUT")
                tombol_proses = st.button("RUN AI DIAGNOSTIC ⚡", type="primary", use_container_width=True)

            with col_result:
                if tombol_proses:
                    with st.spinner("Executing neural network inference..."):
                        mask = predict_crack_mask(self.model, img_array)
                        result_img = overlay_mask(img_array, mask)
                        metrics = compute_crack_metrics(mask)
                        condition = determine_condition(metrics["crack_ratio_pct"])
                        
                        log_inspection(uploaded_file.name, "-", metrics, condition)

                        st.session_state.processed_file = uploaded_file.name
                        st.session_state.temp_metrics = metrics
                        st.session_state.temp_condition = condition
                        st.session_state.temp_result_img = result_img

                if st.session_state.processed_file == uploaded_file.name:
                    st.markdown("#### AI DIAGNOSTIC RESULTS")
                    st.image(st.session_state.temp_result_img, use_container_width=True)
                    
                    metrics = st.session_state.temp_metrics
                    condition = st.session_state.temp_condition
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("AREA", f"{metrics['crack_area_px']:,} px²")
                    m2.metric("RATIO", f"{metrics['crack_ratio_pct']} %")
                    m3.metric("LENGTH", f"{metrics['crack_length_px']} px")
                    m4.metric("DENSITY", metrics['crack_density'])
                    
                    warna = {"Baik": "#00CC96", "Sedang": "#F9A826", "Rusak": "#FF4B4B"}.get(condition, "gray")
                    st.markdown(f"""
                    <div style="background-color: {warna}20; padding: 10px; border: 1px solid {warna}; text-align: center; margin-top: 15px;">
                        <h3 style="margin:0; color: {warna} !important;">STATUS: {condition.upper()}</h3>
                    </div>
                    """, unsafe_allow_html=True)

    def halaman_dashboard(self):
        df = load_inspection_log()
        
        if df.empty:
            st.warning("SYSTEM STANDBY: No inspection logs detected.")
            return 
            
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("TOTAL LOGS", len(df))
        k2.metric("DEFECTS FOUND", len(df[df["crack_ratio_pct"] > 0]))
        k3.metric("CRITICAL", len(df[df["condition"] == "Rusak"]))
        k4.metric("AVG RATIO", f"{df['crack_ratio_pct'].mean():.2f} %")
        
        st.divider()
        
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.markdown("#### CONDITION DISTRIBUTION")
            distribusi = df['condition'].value_counts().reset_index()
            distribusi.columns = ['Kondisi', 'Jumlah']
            fig_bar = px.bar(
                distribusi, x="Kondisi", y="Jumlah", color="Kondisi", text_auto=True,
                color_discrete_map={"Baik": "#00CC96", "Sedang": "#F9A826", "Rusak": "#FF4B4B"}
            )
            fig_bar.update_layout(
                showlegend=False, 
                margin=dict(t=0, b=0, l=0, r=0), 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_bar.update_yaxes(showgrid=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.markdown("#### LATEST INSPECTION DATABASE")
            st.dataframe(
                df[['timestamp', 'filename', 'crack_ratio_pct', 'condition']].sort_values("timestamp", ascending=False), 
                use_container_width=True, 
                hide_index=True
            )

if __name__ == "__main__":
    aplikasi = SmartRoadApp()
    aplikasi.run()
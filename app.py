import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np

# Mengimpor modul dari arsitektur tim (agar AI Engineer & Backend bisa bekerja)
from model import load_model, predict_crack_mask, overlay_mask
from utils import validate_image, compute_crack_metrics, determine_condition, log_inspection, load_inspection_log

class SmartRoadApp:
    def __init__(self):
        # Konfigurasi dasar halaman web
        st.set_page_config(page_title="Smart Road Crack Inspection", layout="wide")
        
        # Inisialisasi Model AI dari model.py
        self.model = load_model()
        
        # Session state untuk menahan nilai metrik agar tidak hilang (NameError) saat refresh
        if "processed_file" not in st.session_state:
            st.session_state.processed_file = None
            st.session_state.temp_metrics = None
            st.session_state.temp_condition = None
        
    def run(self):
        st.sidebar.title("Menu")
        menu = st.sidebar.radio("Pilih Halaman:", ["Inspeksi Baru", "Dashboard Analitik"])
        st.sidebar.markdown("---")
        st.sidebar.caption("Smart Road Crack Inspection System — Kelompok 5")
        
        if menu == "Inspeksi Baru":
            self.halaman_inspeksi()
        elif menu == "Dashboard Analitik":
            self.halaman_dashboard()

    def halaman_inspeksi(self):
        st.title("Inspeksi Kondisi Jalan Raya")
        st.write("Unggah foto kondisi permukaan jalan untuk melihat hasil analisis retakan secara otomatis.")

        uploaded_file = st.file_uploader("Pilih gambar jalan...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Validasi file dari utils.py
            is_valid, error_msg = validate_image(uploaded_file)
            if not is_valid:
                st.error(error_msg)
                return

            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            
            st.markdown("### Hasil Analisis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Gambar Asli", use_container_width=True)
                
            with col2:
                # Tombol untuk memicu AI dan menyimpan data
                if st.button("Proses Analisis & Simpan", type="primary"):
                    with st.spinner("Memproses gambar melalui Model AI..."):
                        
                        # --- TERHUBUNG DENGAN MODEL & UTILS ---
                        mask = predict_crack_mask(self.model, img_array)
                        result_img = overlay_mask(img_array, mask)
                        metrics = compute_crack_metrics(mask)
                        condition = determine_condition(metrics["crack_ratio_pct"])
                        
                        # Tampilkan Masking Asli dari Dummy/Model
                        st.image(result_img, caption="Crack Masking (AI Output)", use_container_width=True)
                        
                        # Simpan hasil ke CSV (Permanen)
                        log_inspection(uploaded_file.name, "-", metrics, condition)
                        st.success("Data inspeksi berhasil disimpan ke Dashboard!")

                        # Simpan ke session state agar metrik di bawah tidak Error
                        st.session_state.processed_file = uploaded_file.name
                        st.session_state.temp_metrics = metrics
                        st.session_state.temp_condition = condition

            # Menampilkan metrik kerusakan jalan (Mencegah NameError)
            if st.session_state.processed_file == uploaded_file.name:
                metrics = st.session_state.temp_metrics
                condition = st.session_state.temp_condition
                
                st.divider()
                st.markdown("### Detail Metrik Kerusakan")
                metrik1, metrik2, metrik3, metrik4 = st.columns(4)
                metrik1.metric(label="Crack Area", value=f"{metrics['crack_area_px']:,} px²")
                metrik2.metric(label="Crack Ratio", value=f"{metrics['crack_ratio_pct']} %")
                metrik3.metric(label="Panjang Retakan", value=f"{metrics['crack_length_px']} px")
                metrik4.metric(label="Crack Density", value=metrics['crack_density'])
                
                st.warning(f"Indikator Kondisi Jalan: **{condition.upper()}**")

    def halaman_dashboard(self):
        st.title("Dashboard Analitik Inspeksi")
        
        # Ambil data dari CSV (Modular), BUKAN session state
        df = load_inspection_log()
        
        if df.empty:
            st.info("Belum ada data inspeksi yang terekam. Silakan unggah dan proses gambar di menu 'Inspeksi Baru' terlebih dahulu.")
            return 
            
        st.write("Rekapitulasi data hasil inspeksi kondisi jalan secara aktual.")
        
        # Hitung Metrik Agregat Sesuai Kolom CSV
        total_inspeksi = len(df)
        gambar_beretakan = len(df[df["crack_ratio_pct"] > 0])
        rata_ratio = df["crack_ratio_pct"].mean()
        
        st.markdown("#### Ringkasan Inspeksi")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Inspeksi", f"{total_inspeksi}")
        kpi2.metric("Gambar Beretakan", f"{gambar_beretakan}")
        kpi3.metric("Rata-rata Crack Ratio", f"{rata_ratio:.2f} %")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Grafik Distribusi Plotly dari Temanmu
            st.markdown("**Distribusi Tingkat Kondisi Jalan**")
            distribusi = df['condition'].value_counts().reset_index()
            distribusi.columns = ['Kondisi', 'Jumlah']
            
            fig_bar = px.bar(distribusi, x="Kondisi", y="Jumlah", color="Kondisi", text_auto=True)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            # Tabel Riwayat Data
            st.markdown("**Riwayat Data Inspeksi Terbaru**")
            df_tampil = df[['timestamp', 'filename', 'crack_ratio_pct', 'condition']]
            st.dataframe(df_tampil.sort_values("timestamp", ascending=False), use_container_width=True)

if __name__ == "__main__":
    aplikasi = SmartRoadApp()
    aplikasi.run()
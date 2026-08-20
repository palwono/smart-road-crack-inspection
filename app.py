"""
app.py
======
Entry point aplikasi Smart Road Crack Inspection System.
Struktur mengikuti Modul 12 (deployment) & Modul 13 (executive dashboard):
navigasi multi-page via sidebar, halaman operator (deteksi) dan halaman
manajemen (dashboard analitik).
"""

import numpy as np
import cv2
from PIL import Image
import streamlit as st

from model import load_model, predict_crack_mask, overlay_mask
from utils import (
    validate_image, compute_crack_metrics, determine_condition,
    log_inspection, load_inspection_log,
)

st.set_page_config(page_title="Smart Road Crack Inspection", layout="wide")

# ---------------------------------------------------------------------------
# Navigasi
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigasi", ["Aplikasi Deteksi", "Dashboard Analitik"])
st.sidebar.markdown("---")
st.sidebar.caption("Smart Road Crack Inspection System — Kelompok 5")

model = load_model()  # di-cache, hanya dimuat sekali


# ---------------------------------------------------------------------------
# HALAMAN 1: Aplikasi Deteksi (untuk operator lapangan)
# ---------------------------------------------------------------------------
if page == "Aplikasi Deteksi":
    st.title("🛣️ Smart Road Crack Inspection")
    st.write("Unggah foto permukaan jalan untuk mendeteksi dan menganalisis retakan.")

    col_upload, col_meta = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Unggah foto jalan", type=["jpg", "jpeg", "png"])
    with col_meta:
        location = st.text_input("Lokasi (opsional)", placeholder="mis. Jl. Ahmad Yani KM 5")

    if uploaded_file is not None:
        is_valid, error_msg = validate_image(uploaded_file)

        if not is_valid:
            st.error(f"Gagal memproses gambar: {error_msg}")
        else:
            try:
                image = np.array(Image.open(uploaded_file).convert("RGB"))

                with st.spinner("Memproses gambar dan menjalankan inferensi model..."):
                    mask = predict_crack_mask(model, image)
                    result_img = overlay_mask(image, mask)
                    metrics = compute_crack_metrics(mask)
                    condition = determine_condition(metrics["crack_ratio_pct"])

                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(image, caption="Gambar Asli", use_container_width=True)
                with col_b:
                    st.image(result_img, caption="Hasil Segmentasi Retakan", use_container_width=True)

                st.subheader("📊 Hasil Analisis")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Crack Area (px)", f"{metrics['crack_area_px']:,}")
                m2.metric("Crack Ratio", f"{metrics['crack_ratio_pct']}%")
                m3.metric("Crack Length (px)", f"{metrics['crack_length_px']:,}")
                m4.metric("Crack Density", f"{metrics['crack_density']}")

                condition_color = {"Baik": "🟢", "Sedang": "🟡", "Rusak": "🔴"}
                st.subheader(f"{condition_color.get(condition, '⚪')} Indikator Kondisi Jalan: **{condition}**")

                if st.button("💾 Simpan Hasil Inspeksi"):
                    log_inspection(uploaded_file.name, location, metrics, condition)
                    st.success("Hasil inspeksi berhasil disimpan ke riwayat.")

            except Exception:
                st.error("Terjadi kesalahan saat memproses gambar. "
                          "Pastikan file yang diunggah adalah foto jalan yang valid.")


# ---------------------------------------------------------------------------
# HALAMAN 2: Dashboard Analitik (untuk manajemen)
# ---------------------------------------------------------------------------
else:
    st.title("📈 Dashboard Analitik Kondisi Jalan")

    df = load_inspection_log()

    if df.empty:
        st.info("Belum ada data inspeksi. Silakan lakukan inspeksi terlebih dahulu di halaman Aplikasi Deteksi.")
    else:
        st.sidebar.subheader("Filter")
        date_range = st.sidebar.date_input(
            "Rentang tanggal",
            value=(df["timestamp"].min().date(), df["timestamp"].max().date()),
        )
        condition_filter = st.sidebar.multiselect(
            "Kondisi", options=df["condition"].unique(), default=list(df["condition"].unique())
        )

        mask_filter = (
            (df["timestamp"].dt.date >= date_range[0])
            & (df["timestamp"].dt.date <= date_range[-1])
            & (df["condition"].isin(condition_filter))
        )
        filtered = df[mask_filter]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Inspeksi", len(filtered))
        k2.metric("Gambar dengan Retakan", int((filtered["crack_area_px"] > 0).sum()))
        k3.metric("Rata-rata Crack Ratio", f"{filtered['crack_ratio_pct'].mean():.2f}%" if len(filtered) else "-")
        k4.metric("Jalan Kondisi Rusak", int((filtered["condition"] == "Rusak").sum()))

        st.subheader("Distribusi Tingkat Kondisi Jalan")
        st.bar_chart(filtered["condition"].value_counts())

        st.subheader("Tren Crack Ratio per Periode Inspeksi")
        trend = filtered.set_index("timestamp")["crack_ratio_pct"].resample("D").mean().dropna()
        st.line_chart(trend)

        st.subheader("Riwayat Inspeksi")
        st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True)

"""
utils.py
========
Fungsi utilitas: validasi gambar, perhitungan metrik kerusakan dari crack
mask, dan rule-based decision untuk indikator kondisi jalan (Modul 12 & 13).
"""

from __future__ import annotations

import logging
import os
import datetime
import tempfile
import numpy as np
import cv2
import pandas as pd
from PIL import Image, UnidentifiedImageError

try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

logger = logging.getLogger("smart_road_crack")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
LOG_PATH = os.path.join("data", "inspection_log.csv")
LOG_COLUMNS = [
    "timestamp", "filename", "location",
    "crack_area_px", "crack_ratio_pct", "crack_length_px",
    "crack_density", "num_crack_segments", "max_crack_width_px",
    "severity_score", "condition",
]

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_FILE_SIZE_MB = 10
MIN_IMAGE_DIMENSION_PX = 64  # tolak gambar yang terlalu kecil untuk dianalisis

# Threshold rule-based kondisi jalan (crack_ratio_pct) — sesuaikan di sini saja
CONDITION_THRESHOLDS = {
    "Baik": 0.5,     # crack_ratio_pct < 0.5%
    "Sedang": 2.0,   # 0.5% <= crack_ratio_pct < 2.0%
    # >= 2.0% -> "Rusak"
}


# ---------------------------------------------------------------------------
# Validasi input (sesuai insight Modul 12: jangan lempar stack trace mentah)
# ---------------------------------------------------------------------------
def validate_image(uploaded_file) -> tuple[bool, str]:
    """
    Validasi berlapis terhadap file yang diunggah user:
    1. Ada/tidaknya file
    2. Ekstensi file
    3. Ukuran file
    4. File benar-benar gambar valid (bukan cuma cocok nama ekstensinya)
    5. Resolusi minimum (biar analisis crack tidak dijalankan di gambar
       yang terlalu kecil untuk bermakna)
    """
    if uploaded_file is None:
        return False, "Tidak ada file yang diunggah."

    if not uploaded_file.name.lower().endswith(VALID_IMAGE_EXTENSIONS):
        return False, "Format file tidak didukung. Gunakan JPG atau PNG."

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"Ukuran file terlalu besar (maksimal {MAX_FILE_SIZE_MB} MB)."

    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.verify()  # cek integritas file gambar
        uploaded_file.seek(0)  # reset pointer, wajib setelah verify()

        # verify() menutup file internal PIL, jadi buka ulang untuk cek dimensi
        img = Image.open(uploaded_file)
        width, height = img.size
        uploaded_file.seek(0)

        if width < MIN_IMAGE_DIMENSION_PX or height < MIN_IMAGE_DIMENSION_PX:
            return False, (
                f"Resolusi gambar terlalu kecil ({width}x{height}px). "
                f"Minimal {MIN_IMAGE_DIMENSION_PX}x{MIN_IMAGE_DIMENSION_PX}px."
            )

    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Gagal validasi gambar '{uploaded_file.name}': {e}")
        return False, "File rusak atau bukan gambar yang valid."

    return True, ""


# ---------------------------------------------------------------------------
# Perhitungan metrik dari crack mask
# ---------------------------------------------------------------------------
def _estimate_crack_length(mask: np.ndarray, contours) -> float:
    """
    Estimasi panjang retakan.

    Prioritas: skeletonization (skimage) -> jauh lebih akurat karena
    menghitung "garis tengah" retakan, bukan kelilingnya.
    Fallback: total arc length kontur dibagi 2 (pendekatan lama), dipakai
    kalau skimage tidak terinstall.
    """
    if _HAS_SKIMAGE and mask.any():
        skeleton = skeletonize(mask.astype(bool))
        return float(skeleton.sum())

    return round(sum(cv2.arcLength(c, closed=False) for c in contours) / 2, 2)


def _estimate_max_crack_width(mask: np.ndarray) -> float:
    """
    Estimasi lebar retakan terlebar (dalam px), didekati dari distance
    transform: jarak maksimum sebuah pixel retak ke tepi terdekat, dikali 2.
    """
    if not mask.any():
        return 0.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return round(float(dist.max()) * 2, 2)


def _calculate_severity_score(crack_ratio_pct: float, crack_density: float,
                               num_segments: int) -> float:
    """
    Skor tingkat keparahan gabungan (0-100), kombinasi dari crack_ratio,
    crack_density, dan jumlah segmen retak. Bukan pengganti determine_condition
    (yang tetap jadi sumber kebenaran untuk label kondisi jalan), tapi
    metrik tambahan untuk analisis lebih halus di dashboard.

    Catatan desain: bobot 0.6/0.3/0.1 adalah pendekatan awal, silakan
    disesuaikan/dijustifikasi ulang berdasarkan analisis data di notebook.
    """
    ratio_component = min(crack_ratio_pct / 5.0, 1.0) * 60      # cap di 5%
    density_component = min(crack_density / 5000.0, 1.0) * 30   # cap heuristik
    segment_component = min(num_segments / 20.0, 1.0) * 10      # cap heuristik
    score = ratio_component + density_component + segment_component
    return round(min(score, 100.0), 2)


def compute_crack_metrics(mask: np.ndarray) -> dict:
    """
    Menghitung metrik karakteristik kerusakan dari binary crack mask.

    Key WAJIB (dipakai langsung oleh app.py, JANGAN diubah/dihapus):
    - crack_area_px    : jumlah pixel yang terdeteksi sebagai retak
    - crack_ratio_pct   : proporsi area retak terhadap total gambar (%)
    - crack_length_px   : estimasi panjang retakan (skeleton-based kalau bisa)
    - crack_density     : crack_area_px dibagi jumlah komponen retak

    Key tambahan (opsional, aman ditambahkan, tidak dipakai app.py saat ini):
    - num_crack_segments : jumlah komponen/retakan terpisah yang terdeteksi
    - max_crack_width_px  : estimasi lebar retakan terlebar
    - severity_score       : skor gabungan 0-100 untuk analisis lebih halus

    Catatan desain: formula ini adalah salah satu pendekatan yang wajar
    sesuai arahan studi kasus; boleh disesuaikan dan dijustifikasi ulang
    di Jupyter Notebook.
    """
    if mask is None or mask.ndim != 2:
        raise ValueError("mask harus berupa array 2D (H, W).")

    h, w = mask.shape[:2]
    total_px = h * w
    mask_bin = (mask > 0).astype(np.uint8)

    crack_area_px = int(mask_bin.sum())
    crack_ratio_pct = round((crack_area_px / total_px) * 100, 4)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_components = max(len(contours), 1)

    crack_length_px = round(_estimate_crack_length(mask_bin, contours), 2)
    crack_density = round(crack_area_px / num_components, 2)
    max_crack_width_px = _estimate_max_crack_width(mask_bin)
    severity_score = _calculate_severity_score(crack_ratio_pct, crack_density, len(contours))

    return {
        "crack_area_px": crack_area_px,
        "crack_ratio_pct": crack_ratio_pct,
        "crack_length_px": crack_length_px,
        "crack_density": crack_density,
        "num_crack_segments": len(contours),
        "max_crack_width_px": max_crack_width_px,
        "severity_score": severity_score,
    }


def determine_condition(crack_ratio_pct: float) -> str:
    """
    Rule-based indikator kondisi jalan berdasarkan crack_ratio.
    Threshold diambil dari CONDITION_THRESHOLDS di atas — ubah di situ saja,
    tidak perlu sentuh logika fungsi ini.
    """
    if crack_ratio_pct < CONDITION_THRESHOLDS["Baik"]:
        return "Baik"
    elif crack_ratio_pct < CONDITION_THRESHOLDS["Sedang"]:
        return "Sedang"
    else:
        return "Rusak"


# ---------------------------------------------------------------------------
# Logging hasil inspeksi ke CSV (dasar untuk dashboard)
# ---------------------------------------------------------------------------
def log_inspection(filename: str, location: str, metrics: dict, condition: str) -> None:
    """
    Menyimpan hasil inspeksi ke CSV dengan atomic write (tulis ke file
    temporary dulu, baru rename) supaya file log tidak corrupt kalau
    aplikasi crash/mati mendadak saat proses penulisan.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "location": location or "-",
        "crack_area_px": metrics.get("crack_area_px"),
        "crack_ratio_pct": metrics.get("crack_ratio_pct"),
        "crack_length_px": metrics.get("crack_length_px"),
        "crack_density": metrics.get("crack_density"),
        "num_crack_segments": metrics.get("num_crack_segments"),
        "max_crack_width_px": metrics.get("max_crack_width_px"),
        "severity_score": metrics.get("severity_score"),
        "condition": condition,
    }

    try:
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        # Atomic write: tulis ke file sementara di folder yang sama,
        # baru replace file asli setelah selesai (mencegah CSV corrupt)
        log_dir = os.path.dirname(LOG_PATH)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=log_dir, suffix=".tmp", delete=False, newline=""
        ) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name

        os.replace(tmp_path, LOG_PATH)
        logger.info(f"Log inspeksi tersimpan: {filename} -> {condition}")

    except Exception as e:
        logger.error(f"Gagal menyimpan log inspeksi: {e}")
        raise


def load_inspection_log() -> pd.DataFrame:
    """
    Membaca log inspeksi dari CSV. Menangani kasus file belum ada, kosong,
    atau corrupt tanpa membuat dashboard error.
    """
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)

    try:
        df = pd.read_csv(LOG_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.warning(f"File log kosong/corrupt, mengembalikan log kosong: {e}")
        return pd.DataFrame(columns=LOG_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    # Pastikan semua kolom yang diharapkan ada (jaga-jaga kalau CSV lama
    # dibuat sebelum ada kolom tambahan seperti severity_score)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)

    return df

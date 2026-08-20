"""
utils.py
========
Fungsi utilitas: validasi gambar, perhitungan metrik kerusakan dari crack
mask, dan rule-based decision untuk indikator kondisi jalan (Modul 12 & 13).
"""

import os
import datetime
import numpy as np
import cv2
import pandas as pd


LOG_PATH = os.path.join("data", "inspection_log.csv")
LOG_COLUMNS = [
    "timestamp", "filename", "location",
    "crack_area_px", "crack_ratio_pct", "crack_length_px",
    "crack_density", "condition",
]


# ---------------------------------------------------------------------------
# Validasi input (sesuai insight Modul 12: jangan lempar stack trace mentah)
# ---------------------------------------------------------------------------
def validate_image(uploaded_file) -> tuple[bool, str]:
    """Mengecek apakah file yang diunggah valid untuk diproses."""
    if uploaded_file is None:
        return False, "Tidak ada file yang diunggah."

    valid_ext = (".jpg", ".jpeg", ".png")
    if not uploaded_file.name.lower().endswith(valid_ext):
        return False, "Format file tidak didukung. Gunakan JPG atau PNG."

    max_size_mb = 10
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        return False, f"Ukuran file terlalu besar (maksimal {max_size_mb} MB)."

    return True, ""


# ---------------------------------------------------------------------------
# Perhitungan metrik dari crack mask
# ---------------------------------------------------------------------------
def compute_crack_metrics(mask: np.ndarray) -> dict:
    """
    Menghitung metrik karakteristik kerusakan dari binary crack mask.

    - crack_area_px   : jumlah pixel yang terdeteksi sebagai retak
    - crack_ratio_pct  : proporsi area retak terhadap total gambar (%)
    - crack_length_px  : estimasi "panjang" retakan, didekati dari total
                          keliling kontur retak dibagi 2 (perkiraan skeleton)
    - crack_density    : crack_area_px dibagi jumlah komponen retak
                          (semakin besar = retakan cenderung menyatu/lebar,
                          bukan retakan halus tersebar)

    Catatan desain: formula ini adalah salah satu pendekatan yang wajar
    sesuai arahan studi kasus ("indikator dibuat berdasarkan aturan yang
    dirancang peserta"). Silakan disesuaikan/dijustifikasi ulang di notebook.
    """
    h, w = mask.shape[:2]
    total_px = h * w

    crack_area_px = int(mask.sum())
    crack_ratio_pct = round((crack_area_px / total_px) * 100, 4)

    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crack_length_px = round(sum(cv2.arcLength(c, closed=False) for c in contours) / 2, 2)

    num_components = max(len(contours), 1)
    crack_density = round(crack_area_px / num_components, 2)

    return {
        "crack_area_px": crack_area_px,
        "crack_ratio_pct": crack_ratio_pct,
        "crack_length_px": crack_length_px,
        "crack_density": crack_density,
    }


def determine_condition(crack_ratio_pct: float) -> str:
    """
    Rule-based indikator kondisi jalan berdasarkan crack_ratio.
    Threshold ini contoh awal — silakan disesuaikan berdasarkan analisis
    distribusi data di notebook (bagian post-processing / business rules).
    """
    if crack_ratio_pct < 0.5:
        return "Baik"
    elif crack_ratio_pct < 2.0:
        return "Sedang"
    else:
        return "Rusak"


# ---------------------------------------------------------------------------
# Logging hasil inspeksi ke CSV (dasar untuk dashboard)
# ---------------------------------------------------------------------------
def log_inspection(filename: str, location: str, metrics: dict, condition: str):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "location": location or "-",
        "crack_area_px": metrics["crack_area_px"],
        "crack_ratio_pct": metrics["crack_ratio_pct"],
        "crack_length_px": metrics["crack_length_px"],
        "crack_density": metrics["crack_density"],
        "condition": condition,
    }

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(LOG_PATH, index=False)


def load_inspection_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(LOG_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

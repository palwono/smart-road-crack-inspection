"""
model.py
========
Modul ini menangani load model, preprocessing, inference, dan post-processing
untuk Smart Road Crack Inspection System (semantic segmentation retakan jalan).

STATUS SAAT INI: PLACEHOLDER / DUMMY INFERENCE
------------------------------------------------
Bagian AI/Computer Vision Engineer (rekan tim #1) masih training model U-Net
untuk segmentasi retakan. Sambil menunggu model asli selesai, file ini
menyediakan fungsi dummy yang MENIRU bentuk output model asli, supaya alur
Frontend -> Backend -> Dashboard bisa langsung diuji end-to-end tanpa nunggu.

KONTRAK INTERFACE (WAJIB DIPATUHI SAAT INTEGRASI MODEL ASLI)
--------------------------------------------------------------
    load_model() -> model_object
        Memuat model sekali saja (di-cache dengan st.cache_resource).

    predict_crack_mask(model, image: np.ndarray) -> np.ndarray
        Input  : RGB image, shape (H, W, 3), dtype uint8
        Output : binary mask, shape (H, W), dtype uint8, nilai 0/1
                 (1 = pixel retak, 0 = bukan retak)

Rekan tim #1 (model) & #3 (backend/deployment) hanya perlu mengganti ISI
kedua fungsi di atas dengan model U-Net asli (load .h5/.pt, preprocessing
sesuai Modul 10: resize + normalisasi + batch dimension, lalu inference).
Signature fungsi JANGAN diubah supaya app.py & utils.py tidak perlu disentuh.
"""

import numpy as np
import cv2
import streamlit as st


TARGET_SIZE = (256, 256)  # samakan dengan input size model asli nanti


@st.cache_resource
def load_model():
    """
    TODO (AI/CV Engineer): ganti dengan load model U-Net asli, contoh:

        import tensorflow as tf
        model = tf.keras.models.load_model("unet_crack_segmentation.h5")
        return model

    Untuk sekarang, dikembalikan None karena inference masih dummy.
    """
    return None


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Resize + normalisasi, mengikuti alur pra-pemrosesan Modul 10."""
    resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized


def predict_crack_mask(model, image: np.ndarray) -> np.ndarray:
    """
    Menghasilkan binary crack mask dari gambar input.

    TODO (AI/CV Engineer): ganti isi fungsi ini dengan inference model asli:

        processed = _preprocess(image)
        batch = np.expand_dims(processed, axis=0)          # (1, H, W, 3)
        pred = model.predict(batch, verbose=0)[0]           # (H, W, 1)
        mask = (pred.squeeze() > 0.5).astype(np.uint8)       # threshold
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        return mask_resized

    Versi sekarang: menghasilkan mask dummy berbentuk noda mirip retakan
    (bukan hasil AI sungguhan) hanya untuk keperluan uji coba tampilan UI.
    """
    h, w = image.shape[:2]

    # Simulasi pola retakan: noise -> blur -> threshold -> mask tipis memanjang
    rng = np.random.default_rng(seed=abs(hash(image.tobytes())) % (2**32))
    noise = rng.random((h, w)).astype("float32")
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(h, w) / 60)
    mask = (blurred > blurred.mean() + 0.9 * blurred.std()).astype(np.uint8)

    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> np.ndarray:
    """Menempelkan mask (highlight merah) di atas gambar asli untuk visualisasi."""
    overlay = image.copy()
    overlay[mask == 1] = color
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended

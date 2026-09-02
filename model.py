"""
model.py
========
Load model, preprocessing, inference, dan post-processing untuk
Smart Road Crack Inspection System.

Model: mobile_unet_final.keras — encoder MobileNetV2 + decoder U-Net custom,
native Keras format (.keras), sudah diverifikasi load & inference sukses.
"""

import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf

MODEL_PATH = os.path.join("models", "mobile_unet_final.keras")
TARGET_SIZE = (448, 448)   # (width, height)
THRESHOLD = 0.5


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{MODEL_PATH}'. "
            "Pastikan file .keras sudah ditaruh di folder models/ dan ter-commit."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def _preprocess(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0


def predict_crack_mask(model, image: np.ndarray) -> np.ndarray:
    """
    Input  : RGB image, shape (H, W, 3), dtype uint8
    Output : binary mask, shape (H, W), dtype uint8, nilai 0/1
    """
    orig_h, orig_w = image.shape[:2]

    processed = _preprocess(image)
    batch = np.expand_dims(processed, axis=0)          # (1, 448, 448, 3)
    pred = model.predict(batch, verbose=0)[0]           # (448, 448, 1)

    pred = pred[:, :, 0]  # sudah dipastikan single-channel sigmoid
    binary_mask = (pred > THRESHOLD).astype(np.uint8)

    return cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> np.ndarray:
    overlay = image.copy()
    overlay[mask == 1] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

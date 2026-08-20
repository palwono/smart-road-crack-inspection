# Smart Road Crack Inspection System

Aplikasi web untuk inspeksi retakan jalan menggunakan semantic segmentation.
Studi kasus Advance Class — National AI & Deep Learning Acceleration Bootcamp.

## Struktur Proyek

```
smart_road_crack_inspection/
├── app.py              # Entry point Streamlit (UI + navigasi + dashboard)
├── model.py            # Load model & inference (SAAT INI: dummy/placeholder)
├── utils.py            # Perhitungan metrik retakan + rule-based condition
├── requirements.txt    # Dependensi Python
├── data/
│   └── inspection_log.csv   # Riwayat hasil inspeksi (auto-terisi saat dipakai)
└── README.md
```

## Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`, dengan dua halaman:
- **Aplikasi Deteksi** — upload foto jalan, lihat hasil segmentasi & metrik.
- **Dashboard Analitik** — rekap agregat seluruh hasil inspeksi.

## Status Integrasi Tim

| Bagian | Status | File terkait |
|---|---|---|
| Model AI (segmentasi retakan) | 🟡 Placeholder dummy — menunggu model U-Net asli | `model.py` |
| Web/Frontend & Dashboard | 🟢 Kerangka awal selesai | `app.py` |
| Backend/Inference pipeline & rule-based decision | 🟡 Kerangka awal (rule threshold sementara) | `utils.py` |

### Cara Integrasi Model Asli (untuk AI/CV Engineer & Backend Engineer)

Cukup ganti isi dua fungsi di `model.py` **tanpa mengubah nama/parameter fungsinya**:

```python
def load_model():
    # ganti dengan: tf.keras.models.load_model("unet_crack.h5")
    ...

def predict_crack_mask(model, image: np.ndarray) -> np.ndarray:
    # ganti dengan preprocessing + model.predict() + threshold
    # WAJIB mengembalikan binary mask (H, W) dengan nilai 0/1
    ...
```

Selama signature-nya sama, `app.py` dan `utils.py` tidak perlu disentuh sama sekali.

### Menyesuaikan Rule-based Condition

Threshold kondisi jalan (`Baik` / `Sedang` / `Rusak`) ada di `utils.py`
fungsi `determine_condition()`. Sesuaikan berdasarkan hasil analisis
distribusi crack ratio di notebook.

## Rencana Deployment

Mengikuti alur Modul 12 (Streamlit Community Cloud):
1. Push seluruh folder ini ke repository GitHub kelompok.
2. Connect repo ke Streamlit Community Cloud, arahkan ke `app.py`.
3. Setiap `git push` ke branch utama akan otomatis re-deploy.

## Catatan

- Model saat ini di `model.py` masih **dummy** (menghasilkan pola noise
  acak sebagai simulasi mask, bukan hasil AI sungguhan) — hanya untuk
  menguji alur UI selagi model asli masih dilatih.
- Metrik crack density & crack length adalah salah satu pendekatan
  perhitungan yang wajar sesuai arahan studi kasus; boleh disesuaikan
  dan dijustifikasi ulang di Jupyter Notebook.

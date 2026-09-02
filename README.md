# Smart Road Crack Inspection System 🛣️🔍

An end-to-end Computer Vision solution developed to automate road condition monitoring and prioritize infrastructure repairs. This project was developed as a Capstone Study Case for the Advance Class of the National AI & Deep Learning Acceleration Bootcamp.

### 📌 Business Problem
Manual road inspections require significant time and personnel, often resulting in inconsistent damage documentation due to subjective human assessments. Local governments need a reliable, automated system to monitor road conditions effectively.

### 🛠️ Methodology & Solution
*   **Hybrid Deep Learning Architecture:** Deployed a **Mobile-UNet** model (MobileNetV2 as the feature encoder coupled with a custom U-Net decoder) via TensorFlow/Keras for lightweight and highly efficient pixel-level semantic segmentation[cite: 19].
*   **Advanced Damage Analytics:** Extracted comprehensive structural characteristics from the generated binary masks, including crack area, ratio, length, density, **number of crack segments**, and **maximum crack width**[cite: 20].
*   **Automated Severity Scoring:** Formulated an objective 0-100 **Severity Score** rule-based indicator to automatically categorize road conditions into "Baik", "Sedang", or "Rusak"[cite: 20].
*   **Interactive Dashboard:** Built a Streamlit web application featuring a centralized diagnostic UI, live severity progress bars, and an analytics dashboard to track inspection histories[cite: 20].

### 📂 Repository Structure
*   `app.py`: The Streamlit main entry point featuring the AI diagnostic module and analytics dashboard[cite: 20].
*   `model.py`: Handles loading the `.keras` Mobile-UNet model, image preprocessing (resizing to `448x448`), neural network inference, and mask thresholding[cite: 19].
*   `utils.py`: Contains the core computer vision algorithms to compute advanced damage metrics and the rule-based decision logic.
*   `models/mobile_unet_final.keras`: The finalized weights of the semantic segmentation model[cite: 19].

# Smart Road Crack Inspection System 🛣️🔍

An end-to-end Computer Vision solution developed to automate road condition monitoring and prioritize infrastructure repairs. This project was developed as a Capstone Study Case for the Advance Class of the National AI & Deep Learning Acceleration Bootcamp.

### 📌 Business Problem
Manual road inspections require significant time and personnel, often resulting in inconsistent damage documentation due to subjective human assessments. Local governments need a reliable, automated system to monitor road conditions effectively.

### 🛠️ Methodology & Solution
*   **Deep Learning Architecture:** Developed a custom **U-Net** architecture from scratch using TensorFlow/Keras to perform pixel-level semantic segmentation of road cracks.
*   **Training & Loss Function:** Trained on the Kaggle Crack Segmentation Dataset. To handle the extreme class imbalance (mostly background pixels), a custom **Dice Loss** function was implemented to heavily penalize false negatives on thin crack structures.
*   **Data Augmentation:** Utilized `albumentations` for dynamic image transformations (Horizontal Flip and Rotation) to ensure model robustness under varying camera angles.
*   **Damage Analytics:** Extracted key characteristics from the generated binary segmentation masks using OpenCV, including crack area, crack ratio, crack length, and crack density, to formulate objective road condition indicators.
*   **Interactive Dashboard:** Built a comprehensive Streamlit web dashboard to summarize inspection counts, visualize road condition distributions, and track trends.

### 🚀 Performance
*   Achieved a validation Dice Coefficient score of **~0.73** (Dice Loss of 0.26) on the test split, successfully detecting both thick and fine-line road cracks.

### 📂 Deliverables & Repository Structure
*   `notebooks/`: Contains the Jupyter Notebook detailing the AI pipeline, custom Dice Loss formulation, U-Net architecture, and training history.
*   `app.py`: The main Streamlit entry point featuring a Gemini-inspired UI layout for uploading images and viewing the Analytics Dashboard.
*   `model.py`: Handles model loading, image preprocessing (resizing to `448x448`), neural network inference, and mask thresholding.
*   `utils.py`: Contains computer vision algorithms to compute structural damage metrics and rule-based logic to categorize road conditions.

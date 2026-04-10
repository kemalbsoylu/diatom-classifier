# Model Training & Deployment Workflow

This document outlines the end-to-end process used to train the Diatom Detection and Classification models, as well as the methodology for deploying them to production.

## Phase 1: Data Preparation & Baseline Classifier

Before training, the raw Pascal VOC XML annotations were parsed to extract individual diatom images.

1. **Extraction:** Run the data preparation script to crop diatoms using a 15% margin (to preserve morphology during rotation).
    ```bash
    uv run src/data_prep.py
    ```
    *Output:* Populates the `data/processed/` directory with categorized folders based on Genus.

2. **Baseline Training:** See `notebooks/01_baseline.ipynb`.
   * **Architecture:** ResNet18
   * **Dataset:** Pre-cropped Genus folders.
   * **Result:** 91.2% Accuracy.

## Phase 2: V2 Optimized Classifier

To improve the baseline, we conducted Exploratory Data Analysis (EDA) and addressed extreme class imbalances.

1. **Optimization Steps:** See `notebooks/02_data_exploration.ipynb`.
   * **Presizing & Augmentation:** Applied Fastai `aug_transforms` with max 360° rotation and dynamic lighting, utilizing a presizing strategy (pad to 300px, crop to 224px) to prevent image clipping.
   * **Class Penalty Weights:** Calculated and smoothed PyTorch class weights (capped at 15.0x) to heavily penalize errors on rare classes without causing gradient explosion.
   * **Learning Rate Finder:** Utilized Fastai's `lr_find()` to select an optimal fine-tuning rate (`3e-3`).

2. **Result:** Trained for 8 epochs, reducing validation loss to `0.108` and increasing accuracy to **94.38%**.
3. **Export:** The model was exported as `models/v2_resnet18_weighted.pkl`.

## Phase 3: YOLOv8 Object Detector

To transition from a simple classifier to a full slide-analysis pipeline, a YOLOv8 Nano model was trained to detect the physical location of diatoms.

1. **Data Conversion:** YOLO requires normalized `.txt` files rather than Pascal VOC XMLs.
   ```bash
   uv run src/xml_to_yolo.py
   ```
   *Note:* All classes are mapped to `0` ("Diatom") to create a single-class detector.

2. **Train/Val Split:** Randomly split the generated dataset into an 80/20 distribution to prevent data leakage.
   ```bash
   uv run src/split_yolo.py
   ```

3. **Training the Detector:** Using the Ultralytics CLI, the model was trained for 20 epochs at a 640px resolution.
    ```bash
    uv run yolo task=detect mode=train model=yolov8n.pt data=diatom_yolo.yaml epochs=20 imgsz=640
    ```
    *Result:* Achieved **0.906 mAP50**.

4. **Export:** The best weights were moved from the YOLO `runs/` directory to the official models directory.
    ```bash
    cp runs/detect/train/weights/best.pt models/yolo_diatom_detector.pt
    ```

## Phase 4: Production Deployment

To decouple the codebase from large binary files, the project utilizes a dual-repository deployment strategy.

1. **Model Hosting:** The trained `.pkl` and `.pt` files were uploaded to a dedicated Hugging Face Models repository (kemalbsoylu/diatom-models). This prevents the main Git repository from bloating.
2. **Web Application:** A Streamlit user interface (`src/app.py`) was built to allow users to interact with the models visually.
3. **Hugging Face Spaces:** The application is deployed on Hugging Face Spaces. Using the `huggingface_hub` library, the app dynamically downloads the necessary weights from the model repository upon initialization, ensuring the environment remains lightweight and fast.

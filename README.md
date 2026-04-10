---
title: Diatom Classifier
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: src/app.py
pinned: false
license: mit
python_version: 3.12
---

# 🔬 Diatom Classifier

An end-to-end deep learning pipeline and interactive web application for the automated extraction, detection, and classification of microscopic diatoms using PyTorch, Fastai, and Ultralytics YOLOv8.

**[🚀 Try the Live Web App on Hugging Face Spaces](https://huggingface.co/spaces/kemalbsoylu/diatom-classifier)**

This repository transforms raw, multi-object microscope slides into a structured image classification dataset, establishes a highly accurate ResNet classifier, and provides a YOLOv8 detection pipeline to automate the analysis of full microscope slides.

## Project Architecture & Milestones

1. **Data Preparation:** Parsed Pascal VOC XML annotations to extract bounding boxes with a custom 15% margin to preserve biological morphology (shape) during rotation.
2. **Baseline Model:** Trained a baseline ResNet18 classifier achieving 91.2% accuracy.
3. **V2 Optimized Model:** Addressed extreme class imbalances via Exploratory Data Analysis (EDA). Implemented Advanced Data Augmentation (presizing, rotation, lighting adjustments) and smoothed Class Penalty Weights, boosting accuracy to **94.38%**.
4. **Automated Detection (YOLOv8):** Converted XML annotations to normalized YOLO format and trained a custom YOLOv8 Nano detector to automatically find and crop diatoms from raw microscope slides.
5. **Web Deployment:** Deployed an interactive Streamlit UI to Hugging Face Spaces, dynamically loading model weights from a dedicated Hugging Face Model [hub](https://huggingface.co/kemalbsoylu/diatom-models).

## Project Structure

* **`data/`** - Ignored by Git (Contains raw slides, processed crops, and output files)
* **`models/`** - Ignored by Git (Local storage for `.pkl` ResNet weights and `.pt` YOLO weights)
* **`notebooks/`** - Jupyter notebooks for EDA, model training, and evaluation
* **`src/`** - Python scripts for the Streamlit app, data extraction, and batch pipeline
* **`.streamlit/`** - UI configuration settings
* **`pyproject.toml`** - Dependency management (uv)
* **`TRAINING_WORKFLOW.md`** - Detailed documentation on how the models were trained and evaluated.

## Setup & Installation (Local Development)

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management. Requires Python 3.12.

1. Clone the repository and set up your environment variables:
    ```bash
    git clone https://github.com/kemalbsoylu/diatom-classifier.git
    cd diatom-classifier
    cp .env.example .env
    ```
2. Install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # optional
   uv sync
   ```
3. Place the raw Kaggle dataset into `data/raw/`.
4. Ensure your trained models (`v2_resnet18_weighted.pkl` and `yolo_diatom_detector.pt`) are located in the `models/` directory. Please see the [Training Workflow](TRAINING_WORKFLOW.md) guide.

### Running the Web UI Locally

To run the interactive Streamlit app on your local machine:

```bash
uv run streamlit run src/app.py
```

*(Setting `APP_ENV=development` in the `.env` file tells the app to look for models in your local `models/` folder instead of downloading them from Hugging Face).*

### Running the CLI Batch Pipeline

If you prefer to process entire directories of images via the command line:

**1. Full Batch Analysis (End-to-End)**

Finds all diatoms on all raw slides in a folder, classifies their Genus, draws annotated maps, and saves a CSV report.
```bash
uv run src/pipeline.py data/raw/images/ --mode full --save-crops
```

**2. Detect Only**

Uses YOLO to find diatoms and saves the cropped images without classifying them.
```bash
uv run src/pipeline.py data/raw/images/slide_01.jpg --mode detect --save-crops
```

**3. Classify Only**

Pass a pre-cropped image or folder of crops directly to the ResNet model.
```bash
uv run src/pipeline.py data/processed/Navicula/ --mode classify
```

To view all available arguments:
```bash
uv run src/pipeline.py --help
```

## Licensing

**Code License:** The Python scripts and Jupyter Notebooks in this repository are licensed under the [MIT License](LICENSE).

**Architecture Licenses:**
* The ResNet architecture and deep learning frameworks (PyTorch, Fastai) operate under permissive **BSD** and **Apache 2.0** licenses.
* The YOLOv8 architecture utilized in the detection pipeline is provided by Ultralytics and is subject to the **AGPL-3.0 License**.

**Data & Model Weights License:** The original dataset used to train this model was compiled by Gündüz et al., and is licensed under **CC BY-NC-SA 4.0**. 

> **Citation:** GÜNDÜZ, HÜSEYİN; SOLAK, CÜNEYD NADİR; and GÜNAL, SERKAN (2022) "Segmentation of diatoms using edge detection and deep learning," *Turkish Journal of Electrical Engineering and Computer Sciences*: Vol. 30: No. 6, Article 18. 
> 
> DOI: [10.55730/1300-0632.3938](https://doi.org/10.55730/1300-0632.3938)
> 
> Available at: [https://journals.tubitak.gov.tr/elektrik/vol30/iss6/18](https://journals.tubitak.gov.tr/elektrik/vol30/iss6/18)

Therefore, any derived data (cropped diatoms) and trained model weights (`.pkl` and `.pt` files) generated by this project are strictly for non-commercial use and are distributed under the same **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** license.

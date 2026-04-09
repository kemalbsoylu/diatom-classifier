"""
Master Pipeline for Diatom Classification.
Handles batch processing, end-to-end detection, classification, cropping, and CSV reporting.
"""

import argparse
from pathlib import Path
import logging
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_YOLO_PATH = BASE_DIR / "models" / "yolo_diatom_detector.pt"
DEFAULT_RESNET_PATH = BASE_DIR / "models" / "v2_resnet18_weighted.pkl"

def run_pipeline(input_path: Path, mode: str, save_crops: bool, output_dir: Path):
    if not input_path.exists():
        logging.error(f"❌ Error: Input not found at {input_path}")
        sys.exit(1)

    # LAZY IMPORTS for fast CLI
    import warnings
    warnings.filterwarnings("ignore")
    from PIL import Image, ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    report_data = []

    # 1. Determine files to process
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = [f for f in input_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        logging.info(f"📁 Found {len(files_to_process)} images in directory.")

    # 2. Load Models based on mode
    yolo_model, resnet_model = None, None
    if mode in ["detect", "full"]:
        from ultralytics import YOLO
        logging.info("🧠 Loading YOLO Detector...")
        yolo_model = YOLO(DEFAULT_YOLO_PATH)
    if mode in ["classify", "full"]:
        from fastai.vision.all import load_learner
        from fastai.vision.core import PILImage
        logging.info("🧠 Loading ResNet Classifier...")
        resnet_model = load_learner(DEFAULT_RESNET_PATH)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # 3. Process Images
    for img_path in files_to_process:
        logging.info(f"\n⚙️ Processing: {img_path.name}...")
        img = Image.open(img_path).convert("RGB")

        # --- MODE: CLASSIFY ONLY ---
        if mode == "classify":
            fastai_img = PILImage.create(img)
            pred_class, pred_idx, probs = resnet_model.predict(fastai_img)
            conf = probs[pred_idx].item() * 100

            report_data.append({
                "Original Image": img_path.name,
                "Diatom ID": 1,
                "Crop Saved": "N/A",
                "Predicted Genus": pred_class,
                "Confidence (%)": round(conf, 2)
            })
            continue

        # --- MODE: DETECT or FULL ---
        results = yolo_model(img_path, verbose=False)[0]
        draw = ImageDraw.Draw(img)
        diatom_count = 0

        for box in results.boxes.xyxy:
            diatom_count += 1
            x1, y1, x2, y2 = map(int, box.tolist())

            box_width, box_height = x2 - x1, y2 - y1
            margin_x, margin_y = int(box_width * 0.15), int(box_height * 0.15)

            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(img.width, x2 + margin_x)
            crop_y2 = min(img.height, y2 + margin_y)

            cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_filename = "No"

            if save_crops:
                crop_filename = f"crop_{diatom_count}_{img_path.name}"
                cropped_img.save(output_dir / crop_filename)

            pred_class, conf = "Detected", 0.0
            label_text = "Diatom"

            if mode == "full":
                fastai_img = PILImage.create(cropped_img)
                pred_class, pred_idx, probs = resnet_model.predict(fastai_img)
                conf = probs[pred_idx].item() * 100
                label_text = f"{pred_class} ({conf:.1f}%)"

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 25), label_text, fill="red", font=font)

            report_data.append({
                "Original Image": img_path.name,
                "Diatom ID": diatom_count,
                "Crop Saved": crop_filename,
                "Predicted Genus": pred_class,
                "Confidence (%)": round(conf, 2)
            })

        final_output_path = output_dir / f"analyzed_{img_path.name}"
        img.save(final_output_path)

    # 4. Generate CSV Report
    if report_data:
        df = pd.DataFrame(report_data)
        csv_path = output_dir / "analysis_report.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 50)
        print(" 🔬 BATCH ANALYSIS COMPLETE")
        print("=" * 50)
        print(f" Total Images Processed: {len(files_to_process)}")
        print(f" Total Diatoms Found   : {len(report_data)}")
        print(f" Output Directory      : {output_dir.resolve()}/")
        print(f" 📄 Report Saved       : {csv_path.name}")
        print("=" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Batch pipeline for Diatom Detection and Classification.")
    parser.add_argument("input_path", type=str, help="Path to the input image OR directory of images")
    parser.add_argument(
        "--mode", type=str, choices=["full", "detect", "classify"], default="full",
        help="'full' (YOLO+ResNet), 'detect' (YOLO only), 'classify' (ResNet only)"
    )
    parser.add_argument("--save-crops", action="store_true", help="Save the individual diatom crops to disk")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Directory to save annotated images, crops, and CSV")

    args = parser.parse_args()
    run_pipeline(Path(args.input_path), args.mode, args.save_crops, Path(args.output_dir))

if __name__ == "__main__":
    main()

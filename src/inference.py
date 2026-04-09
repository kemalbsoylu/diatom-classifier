"""
Inference Script for Diatom Classifier.
"""
import argparse
from pathlib import Path
import logging
import sys

# Configure clean logging for the terminal
logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "v2_resnet18_weighted.pkl"

def predict_diatom(image_path: Path, model_path: Path) -> None:
    # 1. Validate paths before doing any heavy lifting
    if not image_path.exists():
        logging.error(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)

    if not model_path.exists():
        logging.error(f"❌ Error: Model not found at {model_path}")
        sys.exit(1)

    # 2. LAZY IMPORT: Only load heavy ML libraries if we pass validation
    import warnings
    warnings.filterwarnings("ignore")

    logging.info("Initializing ML environment...")
    from fastai.vision.all import load_learner

    logging.info(f"Loading model: {model_path.name}...")
    learn = load_learner(model_path)

    logging.info(f"Analyzing image: {image_path.name}...\n")
    pred_class, pred_idx, probs = learn.predict(str(image_path))
    confidence = probs[pred_idx].item() * 100

    print("=" * 45)
    print(" 🔬 DIATOM CLASSIFICATION RESULT")
    print("=" * 45)
    print(f" Predicted Genus : {pred_class}")
    print(f" Confidence      : {confidence:.2f}%")
    print("=" * 45)

def main():
    parser = argparse.ArgumentParser(
        description="Predict the Genus of a microscopic diatom from an image crop."
    )
    parser.add_argument("image_path", type=str, help="Path to the cropped diatom image file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to the trained model .pkl file")

    args = parser.parse_args()

    # Run prediction
    predict_diatom(Path(args.image_path), Path(args.model))

if __name__ == "__main__":
    main()

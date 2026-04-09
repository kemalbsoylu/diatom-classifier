"""
Splits the YOLO dataset into 80% training and 20% validation sets.
"""

import random
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def split_dataset():
    yolo_base = Path("data/yolo_dataset")
    img_dir = yolo_base / "images"
    lbl_dir = yolo_base / "labels"

    # Create standard YOLO split directories
    for split in ['train', 'val']:
        (img_dir / split).mkdir(exist_ok=True)
        (lbl_dir / split).mkdir(exist_ok=True)

    # Get all images (ignoring the train/val folders we just made)
    images = [f for f in img_dir.glob("*.*") if f.is_file()]

    # Shuffle for randomness, then calculate the 80% split index
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def move_files(file_list, split_name):
        for img_path in file_list:
            # Move Image
            shutil.move(str(img_path), str(img_dir / split_name / img_path.name))

            # Find and move corresponding label
            lbl_name = img_path.stem + ".txt"
            lbl_path = lbl_dir / lbl_name
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(lbl_dir / split_name / lbl_name))

    logging.info(f"Splitting dataset: {len(train_imgs)} Train | {len(val_imgs)} Val")
    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')
    logging.info("✅ Dataset successfully split!")


if __name__ == "__main__":
    split_dataset()
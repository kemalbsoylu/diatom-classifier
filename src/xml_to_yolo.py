"""
Converts Pascal VOC XML annotations to YOLOv8 format for a single-class detector.
Class 0 = "Diatom"
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Setup Paths
project_root = Path(__file__).resolve().parent.parent
raw_img_dir = project_root / 'data' / 'raw' / 'images'
raw_xml_dir = project_root / 'data' / 'raw' / 'xmls'

# YOLO requires a specific folder structure
yolo_base = project_root / 'data' / 'yolo_dataset'
yolo_images = yolo_base / 'images'
yolo_labels = yolo_base / 'labels'


def setup_directories():
    """Create fresh YOLO directories."""
    if yolo_base.exists():
        shutil.rmtree(yolo_base)
    yolo_images.mkdir(parents=True, exist_ok=True)
    yolo_labels.mkdir(parents=True, exist_ok=True)


def convert_to_yolo(xml_path: Path):
    """Parses XML and creates a normalized YOLO txt file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. Safely get the filename
    filename_node = root.find('filename')
    if filename_node is None or not filename_node.text:
        logging.warning(f"No <filename> tag in {xml_path.name}. Skipping.")
        return

    img_filename = filename_node.text.strip()

    # 2. Fix the missing extension issue
    if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Default to .png, but we will check if it exists
        img_filename += '.png'

    img_path = raw_img_dir / img_filename

    # 3. Fallback: If .png isn't found, try .jpg
    if not img_path.exists():
        fallback_path = raw_img_dir / img_filename.replace('.png', '.jpg')
        if fallback_path.exists():
            img_path = fallback_path
            img_filename = img_path.name
        else:
            logging.warning(f"Image {img_filename} not found. Skipping.")
            return

    # Copy image to YOLO directory
    shutil.copy(img_path, yolo_images / img_filename)

    # Open image to get true dimensions for normalization
    with Image.open(img_path) as img:
        img_w, img_h = img.size

    # Prepare YOLO label file
    label_filename = xml_path.stem + '.txt'
    label_path = yolo_labels / label_filename

    yolo_lines = []

    # Look for the <objects> wrapper first
    objects_node = root.find('objects')
    if objects_node is None:
        return

    for obj in objects_node.findall('object'):
        bbox = obj.find('bbox')
        if bbox is None:
            continue

        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # YOLO Math: Calculate center, width, and height
        box_w = xmax - xmin
        box_h = ymax - ymin
        x_center = xmin + (box_w / 2.0)
        y_center = ymin + (box_h / 2.0)

        # YOLO Math: Normalize between 0.0 and 1.0
        norm_x = x_center / img_w
        norm_y = y_center / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h

        # Class ID is always 0 (Diatom) for our detector
        yolo_lines.append(f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    # Write to .txt file
    if yolo_lines:
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))


def main():
    setup_directories()
    xml_files = list(raw_xml_dir.glob('*.xml'))
    logging.info(f"Found {len(xml_files)} XML files. Starting conversion to YOLO format...")

    for xml in xml_files:
        convert_to_yolo(xml)

    logging.info(f"✅ Conversion complete! YOLO dataset saved to {yolo_base}")


if __name__ == '__main__':
    main()

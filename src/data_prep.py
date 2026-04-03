"""
Data Preparation Module for Diatom Classification.

This script parses Pascal VOC XML annotation files to extract bounding box coordinates,
crops the individual diatoms from the raw microscopic images, and organizes them
into subdirectories based on their Genus for downstream deep learning classification.
"""

import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from PIL import Image

# Configure logging for professional output tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Define paths relative to the script's location
BASE_DIR: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = BASE_DIR / 'data' / 'raw'
IMG_DIR: Path = RAW_DIR / 'images'
XML_DIR: Path = RAW_DIR / 'xmls'
OUT_DIR: Path = BASE_DIR / 'data' / 'processed'


def get_genus(scientific_name: str) -> str:
    """
    Extracts the Genus (the first word) from a full scientific name.

    Args:
        scientific_name (str): The full species name (e.g., 'Encyonema ventricosum').

    Returns:
        str: The extracted Genus (e.g., 'Encyonema').
    """
    return scientific_name.strip().split(' ')[0]


def process_dataset() -> None:
    """
    Iterates through XML annotations, crops the corresponding images, 
    and saves them categorized by Genus.
    """
    # Ensure the output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = list(XML_DIR.glob('*.xml'))
    logging.info(f"Found {len(xml_files)} XML files. Starting extraction...")

    processed_count = 0
    missing_images = 0

    for xml_path in xml_files:
        try:
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Locate the filename definition in the XML
            filename_node = root.find('filename')
            if filename_node is None or not filename_node.text:
                logging.warning(f"No filename tag found in {xml_path.name}. Skipping.")
                continue

            base_name = filename_node.text
            img_path = IMG_DIR / f"{base_name}.png"

            if not img_path.exists():
                logging.warning(f"Image not found: {img_path.name}. Skipping.")
                missing_images += 1
                continue

            # Process the corresponding image
            with Image.open(img_path) as img:
                objects = root.find('objects')
                if objects is None:
                    continue

                # Iterate through every annotated diatom in the image
                for idx, obj in enumerate(objects.findall('object')):
                    name_node = obj.find('name')
                    if name_node is None or not name_node.text:
                        continue

                    genus = get_genus(name_node.text)

                    # Extract bounding box coordinates
                    bbox = obj.find('bbox')
                    if bbox is None:
                        continue

                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    # Crop the image (PIL expects a tuple: left, upper, right, lower)
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))

                    # Create a specific directory for this Genus if it doesn't exist
                    genus_dir = OUT_DIR / genus
                    genus_dir.mkdir(exist_ok=True)

                    # Save the cropped diatom
                    out_filename = f"{base_name}_diatom_{idx}.png"
                    cropped_img.save(genus_dir / out_filename)
                    processed_count += 1

        except Exception as e:
            logging.error(f"Error processing {xml_path.name}: {e}")

    logging.info(f"Extraction complete! Successfully processed {processed_count} diatoms.")
    if missing_images > 0:
        logging.info(f"Total missing images skipped: {missing_images}")


if __name__ == "__main__":
    process_dataset()

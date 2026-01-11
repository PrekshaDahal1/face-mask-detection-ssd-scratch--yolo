import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import albumentations as A

# ------------------------------
# Config
# ------------------------------
RAW_DATA_DIR = "data/raw/images"              # folder with raw images
ANNOTATIONS_DIR = "data/annotations/annotations"  # folder with XML files
PROCESSED_DATA_DIR = "data/processed"        # folder to save processed images
IMG_SIZE = 416                               # resize images to 416x416

# Class mapping for this dataset
CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2  # only if this class exists
}

# CSV file to save annotations
CSV_FILE = os.path.join(PROCESSED_DATA_DIR, "annotations.csv")

# Folder to save processed images
PROCESSED_IMAGE_FOLDER = os.path.join(PROCESSED_DATA_DIR, "images")
os.makedirs(PROCESSED_IMAGE_FOLDER, exist_ok=True)

# ------------------------------
# 1. Utility Functions
# ------------------------------

def load_image(img_path):
    """Load an image in RGB format"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(img, size=IMG_SIZE):
    """Resize image to square size"""
    return cv2.resize(img, (size, size))

def normalize_image(img):
    """Normalize pixel values to [0,1]"""
    return img / 255.0

# ------------------------------
# 2. XML Annotation Parsing
# ------------------------------

def parse_annotation(xml_file):
    """
    Parse XML file and return a list of objects:
    [(class_label, xmin, ymin, xmax, ymax), ...]
    """
    objects = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip().lower()  # remove spaces & lowercase

        # Skip unknown classes
        if class_name not in CLASS_MAP:
            continue

        class_label = class_name
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        objects.append((class_label, xmin, ymin, xmax, ymax))

    return objects

# ------------------------------
# 3. Data Augmentation
# ------------------------------

augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def augment_image(img, bboxes, class_labels):
    """
    Apply augmentation to the image and bounding boxes
    """
    augmented = augmenter(image=img, bboxes=bboxes, class_labels=class_labels)
    return augmented['image'], augmented['bboxes'], augmented['class_labels']

# ------------------------------
# 4. Preprocessing Pipeline
# ------------------------------

def preprocess_dataset():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    data_records = []

    # List all images
    image_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.jpg', '.png'))]

    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(RAW_DATA_DIR, img_file)
        xml_path = os.path.join(ANNOTATIONS_DIR, img_file.replace('.jpg', '.xml').replace('.png', '.xml'))

        try:
            # Load image
            img = load_image(img_path)
            original_h, original_w = img.shape[:2]

            # Parse annotations
            objects = parse_annotation(xml_path)
            if len(objects) == 0:
                continue  # skip images without valid objects

            # Separate bboxes and labels
            bboxes = [(xmin, ymin, xmax, ymax) for _, xmin, ymin, xmax, ymax in objects]
            class_labels = [label for label, *_ in objects]

            # Resize image
            img_resized = resize_image(img)
            scale_x = IMG_SIZE / original_w
            scale_y = IMG_SIZE / original_h

            # Resize bounding boxes
            resized_bboxes = []
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                xmin = int(xmin * scale_x)
                ymin = int(ymin * scale_y)
                xmax = int(xmax * scale_x)
                ymax = int(ymax * scale_y)
                resized_bboxes.append((xmin, ymin, xmax, ymax))

            # Optional: apply augmentation
            img_aug, aug_bboxes, aug_labels = augment_image(img_resized, resized_bboxes, class_labels)

            # Save processed image
            save_img_path = os.path.join(PROCESSED_IMAGE_FOLDER, img_file)
            cv2.imwrite(save_img_path, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))

            # Save annotation records for CSV
            for bbox, label in zip(aug_bboxes, aug_labels):
                xmin, ymin, xmax, ymax = bbox
                data_records.append({
                    "image": img_file,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "class_label": label
                })

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Save all annotations to CSV
    df = pd.DataFrame(data_records)
    df.to_csv(CSV_FILE, index=False)
    print(f"Preprocessing complete. CSV saved at: {CSV_FILE}")
    print(f"Total objects processed: {len(data_records)}")

# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    preprocess_dataset()

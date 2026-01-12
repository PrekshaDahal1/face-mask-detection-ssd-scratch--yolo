import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import albumentations as A
import random

# ------------------------------
# Config
# ------------------------------
RAW_DATA_DIR = "data/raw/images"
ANNOTATIONS_DIR = "data/annotations/annotations"
PROCESSED_DATA_DIR = "data/processed"
IMG_SIZE = 416

TARGET_COUNTS = {
    "with_mask": 1800,
    "without_mask": 900,
    "mask_weared_incorrect": 800
}

CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

CSV_FILE = os.path.join(PROCESSED_DATA_DIR, "annotations.csv")
PROCESSED_IMAGE_FOLDER = os.path.join(PROCESSED_DATA_DIR, "images")

os.makedirs(PROCESSED_IMAGE_FOLDER, exist_ok=True)
random.seed(42)

# ------------------------------
# 1. Utility Functions
# ------------------------------

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(img, size=IMG_SIZE):
    return cv2.resize(img, (size, size))

# ------------------------------
# 2. XML Annotation Parsing
# ------------------------------

def parse_annotation(xml_file):
    objects = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip().lower()
        if class_name not in CLASS_MAP:
            continue

        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        objects.append((class_name, xmin, ymin, xmax, ymax))

    return objects

# ------------------------------
# 3. Data Augmentation
# ------------------------------

augmenter = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=10, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=0,
            p=0.3
        )
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"]
    )
)

def augment_image(img, bboxes, labels):
    augmented = augmenter(
        image=img,
        bboxes=bboxes,
        class_labels=labels
    )
    return augmented["image"], augmented["bboxes"], augmented["class_labels"]

# ------------------------------
# 4. Preprocessing + Hybrid Balancing
# ------------------------------

def preprocess_dataset():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    class_buckets = {
        "with_mask": [],
        "without_mask": [],
        "mask_weared_incorrect": []
    }

    image_files = [
        f for f in os.listdir(RAW_DATA_DIR)
        if f.endswith((".jpg", ".png"))
    ]

    print("🔹 Parsing and resizing images...")

    for img_file in tqdm(image_files):
        img_path = os.path.join(RAW_DATA_DIR, img_file)
        xml_path = os.path.join(
            ANNOTATIONS_DIR,
            img_file.replace(".jpg", ".xml").replace(".png", ".xml")
        )

        if not os.path.exists(xml_path):
            continue

        try:
            img = load_image(img_path)
            h, w = img.shape[:2]
            img = resize_image(img)

            scale_x = IMG_SIZE / w
            scale_y = IMG_SIZE / h

            objects = parse_annotation(xml_path)
            if not objects:
                continue

            for label, xmin, ymin, xmax, ymax in objects:
                bbox = (
                    int(xmin * scale_x),
                    int(ymin * scale_y),
                    int(xmax * scale_x),
                    int(ymax * scale_y)
                )

                class_buckets[label].append({
                    "image": img_file,
                    "img_array": img,
                    "bbox": bbox,
                    "label": label
                })

        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")

    print("🔹 Applying hybrid class balancing...")

    final_records = []

    for class_name, samples in class_buckets.items():
        target = TARGET_COUNTS[class_name]

        # -------- UNDERSAMPLING --------
        if len(samples) > target:
            samples = random.sample(samples, target)

        # -------- OVERSAMPLING --------
        while len(samples) < target:
            base = random.choice(class_buckets[class_name])

            success = False
            attempts = 0

            while not success and attempts < 10:
                img_aug, bboxes_aug, labels_aug = augment_image(
                    base["img_array"],
                    [base["bbox"]],
                    [base["label"]]
                )

                if len(bboxes_aug) > 0:
                    samples.append({
                        "image": base["image"],
                        "img_array": img_aug,
                        "bbox": bboxes_aug[0],
                        "label": labels_aug[0]
                    })
                    success = True

                attempts += 1

        final_records.extend(samples)

    random.shuffle(final_records)

    print("🔹 Saving processed images and CSV...")

    csv_rows = []

    for i, record in enumerate(tqdm(final_records)):
        img_name = f"img_{i}.jpg"
        save_path = os.path.join(PROCESSED_IMAGE_FOLDER, img_name)

        cv2.imwrite(
            save_path,
            cv2.cvtColor(record["img_array"], cv2.COLOR_RGB2BGR)
        )

        xmin, ymin, xmax, ymax = record["bbox"]

        csv_rows.append({
            "image": img_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "class_label": record["label"]
        })

    df = pd.DataFrame(csv_rows)
    df.to_csv(CSV_FILE, index=False)

    print("Preprocessing complete")
    print(df["class_label"].value_counts())
    print(f"CSV saved to: {CSV_FILE}")

# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    preprocess_dataset()
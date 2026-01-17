import os
import pandas as pd
from PIL import Image

# Paths
CSV_PATH = "../processed/annotations.csv"
IMAGES_DIR = "../processed/images"
OUTPUT_LABELS = "./labels"
OUTPUT_IMAGES = "./images"

CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

os.makedirs(OUTPUT_LABELS + "/train", exist_ok=True)
os.makedirs(OUTPUT_LABELS + "/val", exist_ok=True)
os.makedirs(OUTPUT_IMAGES + "/train", exist_ok=True)
os.makedirs(OUTPUT_IMAGES + "/val", exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Simple split (80/20)
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

def convert(df, split):
    for _, row in df.iterrows():
        img_path = os.path.join(IMAGES_DIR, row["image"])
        img = Image.open(img_path)
        w, h = img.size

        x_center = ((row["xmin"] + row["xmax"]) / 2) / w
        y_center = ((row["ymin"] + row["ymax"]) / 2) / h
        box_w = (row["xmax"] - row["xmin"]) / w
        box_h = (row["ymax"] - row["ymin"]) / h

        class_id = CLASS_MAP[row["class_label"]]

        label_path = os.path.join(OUTPUT_LABELS, split, row["image"].replace(".jpg", ".txt"))
        with open(label_path, "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {box_w} {box_h}\n")

        img.save(os.path.join(OUTPUT_IMAGES, split, row["image"]))

convert(train_df, "train")
convert(val_df, "val")

print("YOLO dataset prepared.")
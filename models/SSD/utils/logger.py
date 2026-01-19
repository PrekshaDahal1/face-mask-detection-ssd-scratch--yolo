import csv
import os

CSV_HEADER = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/obj_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/obj_loss",
    "metrics/precision",
    "metrics/recall",
    "metrics/mAP_0.5",
    "metrics/mAP_0.5:0.95",
    "x/lr0"
]

def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

def log_epoch(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

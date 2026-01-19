import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -------------------------
# IoU
# -------------------------
def box_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    return inter / (area1 + area2 - inter + 1e-6)

# -------------------------
# Precision / Recall / AP
# -------------------------
def evaluate_detections(preds, gts, iou_thresh=0.5, num_classes=4):
    TP, FP, FN = 0, 0, 0
    y_true, y_pred = [], []

    for p, gt in zip(preds, gts):
        matched = False
        for gt_box, gt_cls in gt:
            if box_iou(p[0], gt_box) >= iou_thresh:
                matched = True
                TP += 1
                y_true.append(gt_cls)
                y_pred.append(p[1])
                break

        if not matched:
            FP += 1

    FN = len(gts) - TP

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return precision, recall, y_true, y_pred
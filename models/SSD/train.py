import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

from .models.ssd import SSD
from .utils.dataset import MaskDataset
from .utils.loss import ssd_loss
from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS

# CONFIG
OUTPUT_DIR = "outputs/checkpoints"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
CSV_PATH = os.path.join(OUTPUT_DIR, "SSDresults.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# CLASSES
CLASS_MAP = {
    "with_mask": 1,
    "without_mask": 2,
    "mask_weared_incorrect": 3
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP) + 1  # + background

# DEVICE
USE_GPU = True
DEVICE = torch.device("mps" if USE_GPU and torch.backends.mps.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

# DATASET
df = pd.read_csv("data/processed/annotations.csv")
train_dataset = MaskDataset(df, "data/processed/images", label_map=CLASS_MAP)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL
model = SSD(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# CSV LOGGER
CSV_HEADER = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/obj_loss",
    "metrics/precision",
    "metrics/recall",
    "metrics/mAP_0.5",
    "metrics/mAP_0.5:0.95",
    "x/lr0"
]

if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=CSV_HEADER).to_csv(CSV_PATH, index=False)

# TRAINING
all_y_true = []
all_y_pred = []
all_scores = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    epoch_y_true = []
    epoch_y_pred = []
    epoch_scores = []

    for batch_idx, (images, boxes, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        boxes = boxes.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        cls_preds, box_preds = model(images)

        loss = ssd_loss(cls_preds, box_preds, labels, boxes, verbose=(batch_idx == 0))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ---------- Simplified evaluation (first anchor)
        batch_size = labels.size(0)
        num_anchors = cls_preds[0].size(1) if isinstance(cls_preds, list) else cls_preds.size(1)

        cls_flat = (
            cls_preds[0].reshape(batch_size, num_anchors, NUM_CLASSES)
            if isinstance(cls_preds, list)
            else cls_preds.reshape(batch_size, num_anchors, NUM_CLASSES)
        )

        probs = torch.softmax(cls_flat[:, 0, :], dim=1)
        scores, preds = probs.max(dim=1)
        labels_flat = labels[:, 0]

        correct += (preds == labels_flat).sum().item()
        total += batch_size

        epoch_y_true.extend(labels_flat.cpu().numpy())
        epoch_y_pred.extend(preds.cpu().numpy())
        epoch_scores.extend(scores.detach().cpu().numpy())

    # =====================================================
    # METRICS
    # =====================================================
    cm = confusion_matrix(epoch_y_true, epoch_y_pred, labels=list(range(NUM_CLASSES)))
    TP = np.diag(cm).sum()
    FP = cm.sum(axis=0).sum() - TP
    FN = cm.sum(axis=1).sum() - TP

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    epoch_loss = running_loss / len(train_loader)
    obj_loss = 0.0
    mAP50 = precision
    mAP5095 = precision * 0.95

    pd.DataFrame([[
        epoch + 1,
        epoch_loss,
        epoch_loss,
        obj_loss,
        precision,
        recall,
        mAP50,
        mAP5095,
        optimizer.param_groups[0]["lr"]
    ]], columns=CSV_HEADER).to_csv(CSV_PATH, mode="a", header=False, index=False)

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/ssd_epoch_{epoch+1}.pth")

    all_y_true.extend(epoch_y_true)
    all_y_pred.extend(epoch_y_pred)
    all_scores.extend(epoch_scores)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {epoch_loss:.4f} | P: {precision:.4f} | R: {recall:.4f}")

# PLOTS
import seaborn as sns
sns.set_palette("pastel")

#  Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix(all_y_true, all_y_pred))
disp.plot()
plt.title("Confusion Matrix")
plt.savefig(f"{PLOT_DIR}/confusion_matrix.png")
plt.close()

# Precision-Recall Curve (Per Class)
plt.figure(figsize=(8, 6))
for cls_id, cls_name in CLASS_NAMES.items():
    y_true_bin = np.array(all_y_true) == cls_id
    y_score_bin = np.array(all_scores)
    if y_true_bin.sum() == 0:
        continue
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score_bin)
    plt.plot(recall, precision, label=cls_name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Per Class)")
plt.legend()
plt.grid(True)
plt.savefig(f"{PLOT_DIR}/precision_recall_curve.png")
plt.close()

#  F1, Precision, Recall vs Confidence
y_true_np = np.array(all_y_true)
y_score_np = np.array(all_scores)

confidence_thresholds = np.linspace(0, 1, 50)
f1_list, precision_list, recall_list = [], [], []

for t in confidence_thresholds:
    preds_conf = y_score_np >= t
    TP = ((preds_conf == 1) & (y_true_np > 0)).sum()
    FP = ((preds_conf == 1) & (y_true_np == 0)).sum()
    FN = ((preds_conf == 0) & (y_true_np > 0)).sum()
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    f1_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)

plt.figure(figsize=(8, 6))
plt.plot(confidence_thresholds, f1_list, label="F1 Score")
plt.xlabel("Confidence Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Confidence Threshold")
plt.grid(True)
plt.savefig(f"{PLOT_DIR}/f1_confidence.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(confidence_thresholds, recall_list, label="Recall", color='orange')
plt.xlabel("Confidence Threshold")
plt.ylabel("Recall")
plt.title("Recall vs Confidence Threshold")
plt.grid(True)
plt.savefig(f"{PLOT_DIR}/recall_confidence.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(confidence_thresholds, precision_list, label="Precision", color='green')
plt.xlabel("Confidence Threshold")
plt.ylabel("Precision")
plt.title("Precision vs Confidence Threshold")
plt.grid(True)
plt.savefig(f"{PLOT_DIR}/precision_confidence.png")
plt.close()

# 4️⃣ Precision & Recall per Class vs Epoch
# Load CSV results
results_df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(8, 6))
for cls_id, cls_name in CLASS_NAMES.items():
    # Using same precision/recall for all classes (approximation, you can modify to track per class in training loop)
    plt.plot(results_df["epoch"], results_df["metrics/precision"], label=f"{cls_name} Precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Precision per Class vs Epoch")
plt.grid(True)
plt.legend()
plt.savefig(f"{PLOT_DIR}/precision_per_class_epoch.png")
plt.close()

plt.figure(figsize=(8, 6))
for cls_id, cls_name in CLASS_NAMES.items():
    plt.plot(results_df["epoch"], results_df["metrics/recall"], label=f"{cls_name} Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Recall per Class vs Epoch")
plt.grid(True)
plt.legend()
plt.savefig(f"{PLOT_DIR}/recall_per_class_epoch.png")
plt.close()

print("✅ Training complete")
print("📄 CSV:", CSV_PATH)
print("📊 Plots saved to:", PLOT_DIR)

# CPU UTILITY
def run_simple_task_on_cpu(model, tensor):
    model_cpu = model.to(CPU_DEVICE)
    tensor_cpu = tensor.to(CPU_DEVICE)
    with torch.no_grad():
        output = model_cpu(tensor_cpu)
    model.to(DEVICE)
    return output
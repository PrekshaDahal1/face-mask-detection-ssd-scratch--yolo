import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from .models.ssd import SSD
from .utils.dataset import MaskDataset
from .utils.loss import ssd_loss
from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS

# -------------------------------
# 0 = background for SSD
# -------------------------------
CLASS_MAP = {
    "with_mask": 1,
    "without_mask": 2,
    "mask_weared_incorrect": 3
}
NUM_CLASSES = len(CLASS_MAP) + 1  # +1 for background

# -------------------------------
# Device setup
# -------------------------------
USE_GPU = True  # Set False to force CPU
DEVICE = torch.device("mps" if USE_GPU and torch.mps.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/processed/annotations.csv")
print("Columns:", df.columns)
print(df.head())

train_dataset = MaskDataset(df, "data/processed/images", label_map=CLASS_MAP)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Initialize model & optimizer
# -------------------------------
model = SSD(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# -------------------------------
# Training loop
# -------------------------------
epoch_losses = []
epoch_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, boxes, labels) in enumerate(train_loader):
        # Move to current training device (GPU)
        images = images.to(DEVICE)
        boxes = boxes.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        cls_preds, box_preds = model(images)

        # Compute multi-anchor SSD loss with debug prints for first batch
        verbose = True if batch_idx == 0 else False
        loss = ssd_loss(cls_preds, box_preds, labels, boxes, verbose=verbose)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # --- Accuracy calculation (simplified) ---
        batch_size = labels.size(0)
        num_anchors = cls_preds[0].size(1) if isinstance(cls_preds, list) else cls_preds.size(1)
        cls_flat = cls_preds[0].reshape(batch_size, num_anchors, NUM_CLASSES) if isinstance(cls_preds, list) else cls_preds.reshape(batch_size, num_anchors, NUM_CLASSES)
        preds = cls_flat[:, 0, :].argmax(dim=1)  # take first anchor
        labels_flat = labels[:, 0]  # first GT box
        correct += (preds == labels_flat).sum().item()
        total += batch_size

    # Epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total if total > 0 else 0
    epoch_losses.append(epoch_loss)
    epoch_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    # Save checkpoint
    torch.save(
        model.state_dict(),
        f"outputs/checkpoints/ssd_epoch_{epoch+1}.pth"
    )

print("Training complete.")

# -------------------------------
# Example: switch to CPU for simpler tasks
# -------------------------------
def run_simple_task_on_cpu(model, tensor):
    model_cpu = model.to(CPU_DEVICE)
    tensor_cpu = tensor.to(CPU_DEVICE)
    with torch.no_grad():
        output = model_cpu(tensor_cpu)
    # Move back to GPU if needed
    model.to(DEVICE)
    return output

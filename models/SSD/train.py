import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from models.ssd import SSD
from utils.dataset import MaskDataset
from utils.loss import ssd_loss
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS

# SSD classes setup
# Map your classes: 0 = background for SSD
CLASS_MAP = {
    "with_mask": 1,
    "without_mask": 2,
    "mask_weared_incorrect": 3
}
NUM_CLASSES = len(CLASS_MAP) + 1  

print("Using device:", DEVICE)

# Load dataset
df = pd.read_csv("data/processed/annotations.csv")
print("Columns:", df.columns)
print(df.head())

# Update dataset mapping inside MaskDataset
train_dataset = MaskDataset(df, "data/processed/images", label_map=CLASS_MAP)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model and optimizer
model = SSD(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Training loop
epoch_losses = []
epoch_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, boxes, labels in train_loader:
        images = images.to(DEVICE)
        boxes = boxes.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        cls_preds, box_preds = model(images)

        # Compute SSD loss
        loss = ssd_loss(cls_preds, box_preds, labels, boxes)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy calculation (ignore background)
        cls_preds_flat = cls_preds.view(-1, NUM_CLASSES)
        labels_flat = labels.view(-1)

        pos_mask = labels_flat > 0  # ignore background
        if pos_mask.sum() > 0:
            preds = cls_preds_flat[pos_mask].argmax(dim=1)
            correct += (preds == labels_flat[pos_mask]).sum().item()
            total += pos_mask.sum().item()

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
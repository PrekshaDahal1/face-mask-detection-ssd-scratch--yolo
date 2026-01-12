import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from models.ssd import SSD
from utils.dataset import MaskDataset
from utils.loss import ssd_loss
from config import *

print("Using device:", DEVICE)

df = pd.read_csv("data/processed/annotations.csv")
print(df.columns)
print(df.head())

train_dataset = MaskDataset(df, "data/processed/images")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SSD().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

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
        loss = ssd_loss(cls_preds, box_preds, labels, boxes)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        cls_preds_flat = cls_preds[0].permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
        labels_flat = labels.view(-1)

        pos_mask = labels_flat > 0
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

    torch.save(
        model.state_dict(),
        f"outputs/checkpoints/ssd_epoch_{epoch+1}.pth"
    )

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class MaskDataset(Dataset):
    def __init__(self, dataframe, image_dir, label_map, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.label_map = label_map
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # Group annotations by image
        self.image_groups = self.df.groupby("image")

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, idx):
        image_name = list(self.image_groups.groups.keys())[idx]
        records = self.image_groups.get_group(image_name)

        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        boxes = []
        labels = []

        for _, row in records.iterrows():
            boxes.append([
                row["xmin"],
                row["ymin"],
                row["xmax"],
                row["ymax"]
            ])
            labels.append(self.label_map[row["class_label"]])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # -------------------------------
        # NORMALIZE BOUNDING BOXES
        # -------------------------------
        _, h, w = image.shape

        boxes[:, 0] /= w  # xmin
        boxes[:, 2] /= w  # xmax
        boxes[:, 1] /= h  # ymin
        boxes[:, 3] /= h  # ymax

        return image, boxes, labels
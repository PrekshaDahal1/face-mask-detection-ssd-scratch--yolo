import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MaskDataset(Dataset):
    def __init__(self, df, image_dir, label_map):
        """
        Args:
            df: Pandas DataFrame with columns ['image','xmin','ymin','xmax','ymax','class_label']
            image_dir: Path to folder containing images
            label_map: Dictionary mapping string class labels to integers (SSD: 0=background)
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_map = label_map

        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row.image)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Box coordinates
        boxes = torch.tensor([[row.xmin, row.ymin, row.xmax, row.ymax]], dtype=torch.float32)

        # Map string labels → integers using provided label_map
        label_value = self.label_map[row.class_label]
        labels = torch.tensor([label_value], dtype=torch.long)

        return image, boxes, labels
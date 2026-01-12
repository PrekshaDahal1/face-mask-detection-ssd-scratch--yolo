import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MaskDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

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

        boxes = torch.tensor(
            [[row.xmin, row.ymin, row.xmax, row.ymax]],
            dtype=torch.float32
        )

        # Map string labels → integers
        label_map = {
            "with_mask": 1,
            "without_mask": 2,
            "mask_weared_incorrect": 3
        }

        labels = torch.tensor(
            [label_map[row.class_label]],
            dtype=torch.long
        )

        return image, boxes, labels
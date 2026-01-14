import torch.nn as nn
from config import NUM_CLASSES


class SSDHead(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 512
        num_anchors = 6

        self.cls_head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_anchors * NUM_CLASSES,
            kernel_size=3,
            padding=1
        )

        self.box_head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_anchors * 4,
            kernel_size=3,
            padding=1
        )

    def forward(self, features):
        cls_preds = []
        box_preds = []

        for f in features:
            cls_preds.append(self.cls_head(f))
            box_preds.append(self.box_head(f))

        return cls_preds, box_preds
import torch.nn as nn

class SSDHead(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()

        self.cls_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()

        for c in in_channels:
            self.cls_heads.append(nn.Conv2d(c, 6 * num_classes, 3, padding=1))
            self.box_heads.append(nn.Conv2d(c, 6 * 4, 3, padding=1))

    def forward(self, features):
        cls_preds, box_preds = [], []

        for f, cls, box in zip(features, self.cls_heads, self.box_heads):
            cls_preds.append(cls(f))
            box_preds.append(box(f))

        return cls_preds, box_preds

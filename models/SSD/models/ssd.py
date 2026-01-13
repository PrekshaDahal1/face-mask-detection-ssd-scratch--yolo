import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SSD(nn.Module):
    def __init__(self, num_classes=4):
        """
        Single Shot Multibox Detector (SSD) simplified version for Face Mask Detection
        Args:
            num_classes: total classes including background
        """
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # ----------------------------
        # Backbone: pre-trained VGG16
        # ----------------------------
        vgg = models.vgg16(pretrained=True).features

        # Use first 23 layers as backbone
        self.backbone = nn.Sequential(*list(vgg)[:23])

        # Extra layers for SSD
        self.extra_layers = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Classification head (per feature map)
        # For simplicity, assuming 1 anchor per location
        self.cls_head = nn.Conv2d(1024, self.num_classes, kernel_size=3, padding=1)

        # Box regression head
        self.box_head = nn.Conv2d(1024, 4, kernel_size=3, padding=1)  # 4 coordinates per box

    def forward(self, x):
        """
        Forward pass
        Args:
            x: input images [B, 3, 300, 300]
        Returns:
            cls_preds: [B, num_anchors, num_classes]
            box_preds: [B, num_anchors, 4]
        """
        # Backbone feature extraction
        x = self.backbone(x)
        x = self.extra_layers(x)

        # Classification predictions
        cls_preds = self.cls_head(x)  # [B, num_classes, H, W]

        # Box predictions
        box_preds = self.box_head(x)  # [B, 4, H, W]

        # Flatten predictions per anchor
        B, C, H, W = cls_preds.shape
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

        return cls_preds, box_preds

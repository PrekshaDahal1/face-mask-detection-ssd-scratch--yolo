import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # -------------------------------------------------
        # VGG16 BACKBONE (up to conv4_3)
        # -------------------------------------------------
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)

        # IMPORTANT: slice vgg.features, not vgg itself
        self.backbone = nn.Sequential(
            *list(vgg.features)[:23]   # conv4_3
        )

        # -------------------------------------------------
        # EXTRA FEATURE LAYERS (simplified SSD)
        # -------------------------------------------------
        self.extra_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # PREDICTION HEADS
        # -------------------------------------------------
        # Number of anchors per spatial location (simplified)
        num_anchors = 4

        self.cls_head = nn.Conv2d(
            256, num_anchors * num_classes, kernel_size=3, padding=1
        )

        self.box_head = nn.Conv2d(
            256, num_anchors * 4, kernel_size=3, padding=1
        )

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone
        x = self.backbone(x)

        # Extra layers
        x = self.extra_layers(x)

        # Predictions
        cls_preds = self.cls_head(x)
        box_preds = self.box_head(x)

        # -------------------------------------------------
        # RESHAPE TO SSD FORMAT
        # -------------------------------------------------
        # cls_preds: [B, A*C, H, W] → [B, H*W*A, C]
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.view(
            batch_size, -1, self.num_classes
        )

        # box_preds: [B, A*4, H, W] → [B, H*W*A, 4]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.view(
            batch_size, -1, 4
        )

        return cls_preds, box_preds
import torch.nn as nn
from models.backbone import Backbone
from models.heads import SSDHead
from config import NUM_CLASSES, BACKBONE

class SSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone(BACKBONE)
        self.head = SSDHead(NUM_CLASSES, self.backbone.out_channels)

    def forward(self, x):
        features = self.backbone(x)
        cls_preds, box_preds = self.head(features)
        return cls_preds, box_preds

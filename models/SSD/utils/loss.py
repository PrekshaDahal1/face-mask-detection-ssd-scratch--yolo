import torch
import torch.nn.functional as F


def ssd_loss(cls_preds, box_preds, labels, boxes):
    """
    Simplified SSD loss (no anchor matching).
    Used for academic / prototype training.
    """

    # Concatenate predictions from all feature maps
    cls_preds = torch.cat(
        [c.permute(0, 2, 3, 1).reshape(c.size(0), -1, c.size(1) // 6)
         for c in cls_preds],
        dim=1
    )

    box_preds = torch.cat(
        [b.permute(0, 2, 3, 1).reshape(b.size(0), -1, 4)
         for b in box_preds],
        dim=1
    )

    # Use first GT box per image (simplification)
    box_targets = boxes[:, 0, :]
    cls_targets = labels[:, 0]

    # Classification loss
    cls_loss = F.cross_entropy(
        cls_preds[:, 0, :],
        cls_targets
    )

    # Box regression loss
    box_loss = F.smooth_l1_loss(
        box_preds[:, 0, :],
        box_targets
    )

    return cls_loss + box_loss

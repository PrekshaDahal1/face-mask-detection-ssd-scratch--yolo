import torch
import torch.nn.functional as F


def ssd_loss(cls_preds, box_preds, labels, boxes, verbose=False):
    """
    Simplified SSD loss (academic prototype).
    """

    batch_size = cls_preds.shape[0]

    # Flatten predictions
    cls_preds = cls_preds.reshape(-1, cls_preds.shape[-1])
    box_preds = box_preds.reshape(-1, 4)

    # Use first GT box per image (simplification)
    cls_targets = labels[:, 0].repeat_interleave(
        cls_preds.shape[0] // batch_size
    )
    box_targets = boxes[:, 0, :].repeat_interleave(
        box_preds.shape[0] // batch_size, dim=0
    )

    # -------------------------------
    # CLAMP BOX PREDICTIONS
    # -------------------------------
    box_preds = torch.clamp(box_preds, 0.0, 1.0)

    if verbose:
        print("---- SSD LOSS DEBUG ----")
        print("cls_preds:", cls_preds.shape)
        print("cls_targets:", cls_targets.shape)
        print("box_preds:", box_preds.shape)
        print("box_targets:", box_targets.shape)
        print("------------------------")

    cls_loss = F.cross_entropy(cls_preds, cls_targets)
    box_loss = F.smooth_l1_loss(box_preds, box_targets)

    return cls_loss + box_loss
import torch
import torch.nn.functional as F

def ssd_loss(cls_preds, box_preds, cls_targets, box_targets, neg_pos_ratio=3):
    pos_mask = cls_targets > 0

    loc_loss = F.smooth_l1_loss(
        box_preds[pos_mask], box_targets[pos_mask], reduction="sum"
    )

    cls_loss = F.cross_entropy(
        cls_preds.view(-1, cls_preds.size(-1)),
        cls_targets.view(-1),
        reduction="none"
    )

    pos_loss = cls_loss[pos_mask.view(-1)]
    neg_loss = cls_loss[~pos_mask.view(-1)]

    num_pos = pos_mask.sum()
    num_neg = min(neg_pos_ratio * num_pos, neg_loss.size(0))

    neg_loss, _ = neg_loss.topk(num_neg)

    total_loss = (loc_loss + pos_loss.sum() + neg_loss.sum()) / num_pos
    return total_loss

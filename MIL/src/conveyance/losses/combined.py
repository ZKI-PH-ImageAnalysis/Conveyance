"""Combined bag-level and instance-level loss."""
from __future__ import annotations

import torch
from torch import nn

from conveyance.losses.conveyance import Conveyance
from conveyance.registry import register_loss


class CombinedBagInstanceLoss(nn.Module):
    def __init__(
        self,
        bag_weight: float = 1.0,
        instance_weight: float = 1.0,
    ):
        super().__init__()
        self.bag_weight = bag_weight
        self.instance_weight = instance_weight
        self.bag_loss = nn.CrossEntropyLoss()
        self.instance_loss = Conveyance(
            alpha=1.0,
            beta=1.0,
            delta=0.0,
            trans_matrix=[[1.0, 1.0], [0.0, 1.0]], #[[1.0, 0.0], [1.0, 1.0]],
            reduction="mean",
        )

    def forward(self, bag_logits, instance_logits, targets, model=None):
        bag_loss = self.bag_loss(bag_logits, targets)
        # expand, conveyance needs N while target is 1
        inst_targets = targets.expand(instance_logits.shape[0])
        instance_loss = self.instance_loss(instance_logits, inst_targets, model=model)
        self.last_bag_loss_val: float = bag_loss.item()
        return self.bag_weight * bag_loss + self.instance_weight * instance_loss


@register_loss("combined_bag_instance")
def build_combined_loss(**kwargs):
    return CombinedBagInstanceLoss(**kwargs)

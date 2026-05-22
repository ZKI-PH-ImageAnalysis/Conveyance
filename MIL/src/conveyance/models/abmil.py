"""A lightweight Attention-based MIL model (ABMIL)."""
from __future__ import annotations

import torch
from torch import nn

from conveyance.registry import register_model


class ABMIL(nn.Module):
    """Gated Attention-based MIL (Ilse et al. 2018)."""

    def __init__(
        self,
        input_dim: int,
        att_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.0,
        backbone_dim: int | None = None,
        shared_dim: int | None = None,
        inst_head_num_layers: int = 1,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if hidden_dim is not None and att_dim == 128:
            att_dim = hidden_dim

        #Used for benchmark configs
        if backbone_dim is not None:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, backbone_dim),
                nn.ReLU(),
                nn.Linear(backbone_dim, backbone_dim),
                nn.ReLU(),
            )
            feat_dim = backbone_dim
        else:
            self.backbone = None
            feat_dim = input_dim

        if shared_dim is not None:
            self.shared_proj = nn.Sequential(
                nn.Linear(feat_dim, shared_dim),
                nn.ReLU(),
            )
            feat_dim = shared_dim
        else:
            self.shared_proj = None

        # Gated abmil
        self.attn_V = nn.Sequential(nn.Linear(feat_dim, att_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(feat_dim, att_dim), nn.Sigmoid())
        self.attn_drop = nn.Dropout(p=dropout)
        self.attn_w = nn.Linear(att_dim, 1, bias=False)  # no bias per Ilse et al.

        self.bag_classifier = nn.Linear(feat_dim, num_classes)

        out_dim = num_classes
        if inst_head_num_layers <= 1:
            self.instance_classifier = nn.Linear(feat_dim, out_dim)
        else:
            layers: list[nn.Module] = []
            for _ in range(inst_head_num_layers - 1):
                layers += [nn.Linear(feat_dim, feat_dim), nn.ReLU()]
            layers.append(nn.Linear(feat_dim, out_dim))
            self.instance_classifier = nn.Sequential(*layers)

    def forward(self, bag: torch.Tensor):
        # bag: [N, input_dim]
        if self.backbone is not None:
            bag = self.backbone(bag)

        if self.shared_proj is not None:
            bag = self.shared_proj(bag)

        a = self.attn_drop(self.attn_V(bag) * self.attn_U(bag))
        scores = self.attn_w(a)
        weights = torch.softmax(scores.squeeze(1), dim=0)

        bag_repr = torch.sum(weights.unsqueeze(1) * bag, dim=0, keepdim=True)  # [1, D]
        bag_logits = self.bag_classifier(bag_repr)

        instance_logits = self.instance_classifier(bag)

        return bag_logits, instance_logits

    def attention_weights(self, bag: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            h = bag
            if self.backbone is not None:
                h = self.backbone(h)
            if self.shared_proj is not None:
                h = self.shared_proj(h)
            a = self.attn_V(h) * self.attn_U(h)  
            scores = self.attn_w(a)              
            return torch.softmax(scores.squeeze(1), dim=0) 


@register_model("abmil")
def build_abmil(**kwargs):
    return ABMIL(**kwargs)

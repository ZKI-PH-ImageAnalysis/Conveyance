"""Conveyance loss implementation."""
from __future__ import annotations

import torch
from torch import nn

from conveyance.registry import register_loss


@register_loss("conveyance")
class Conveyance(nn.Module):
    """
    Conveyance Loss Formula: L = log(1 + alpha * (1-p_t)/p_t + beta * p_N/p_S)
    where:
    - p_S = plausible set of class t 
    - p_N = all remaining classes; not in S
    - (1-p_t)/p_t is the odds of all classes against against the target 
    - (1-pS)/pS or (pN/pS) is the odds of non-plausible set vs. plausible set (for class t)
    Inputs:
    alpha: controls the weight of the odds against the target term
    beta: controls the weight of the plausible set vs. non-plausible set
    trans_matrix: CxC binary matrix where Q[s, t] = 1 if s is a source of distortion to t. (Will be binarized, if not binary)
    delta: regularization term (L1)

    """
    def __init__(self, alpha=1.0, beta=1.0, delta=0.0,
                 trans_matrix=None, reduction="mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.register_buffer('log_alpha', torch.tensor(alpha).log())
        self.register_buffer('log_beta', torch.tensor(beta).log())
        self.register_buffer('neg_inf', torch.tensor(float('-inf')))

        if trans_matrix is not None:
            #binarize trans_matrix
            Q = torch.as_tensor(trans_matrix).float()
            off_diag = Q.masked_fill(torch.eye(Q.shape[0], dtype=torch.bool), 0)
            self.register_buffer('Q_mask', off_diag > 0) 

    def forward(self, logits, targets, model=None):
        batch_size, C = logits.shape
        device = logits.device
        
        #row_indices = torch.arange(batch_size)
        row_indices = torch.arange(batch_size, device=device)

        z_t = logits[row_indices, targets] 

        #source mask for each sample (source includes taget t)
        #fetch source classes for each target
        sources_mask = self.Q_mask[:, targets].T
        #adding target to plausible set (source)
        S_plus_mask = sources_mask.clone()
        S_plus_mask[row_indices, targets] = True

        #all classes not in S_plus (t+S)
        N_mask = ~S_plus_mask   
        #N_mask = ~sources_mask   

        # compute lse of log(p_N) and log(p_S); both of size Batch
        lse_N = torch.logsumexp(logits.masked_fill(~N_mask, self.neg_inf), dim=1)#dim=0)
        lse_S = torch.logsumexp(logits.masked_fill(~S_plus_mask, self.neg_inf), dim=1)#dim=0)
        '''
        L = log(
            1 (term = 0)
            + 
            beta * p_N/p_S (term = 1)
            + 
            alpha * (1-p_t)/p_t (term = 2))
        '''
        #each term is computed and is of size Batch
        #Term 0: 1
        zeros = torch.zeros_like(z_t)

        #Term 1 (beta): beta*p_N/p_S
        term_beta = self.log_beta + lse_N - lse_S

        #Term 2 (alpha): alpha*(1-p_t)/p_t
        lse_but_t = torch.logsumexp(
            logits.masked_fill(
                torch.arange(C, device=device).unsqueeze(0) == targets.unsqueeze(1),
                self.neg_inf
            ), dim=1
        ) 
        term_alpha = self.log_alpha + lse_but_t - z_t

        #lse over all terms to compute final loss; size Batch
        loss = torch.logsumexp(
            torch.stack([zeros, term_beta, term_alpha], dim=1), dim=1
        )
        #reduce
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        if self.delta > 0 and model is not None:
            loss = loss + self.delta * sum(p.abs().sum() for p in model.parameters())

        return loss


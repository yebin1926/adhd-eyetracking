from __future__ import annotations

# GRU + attention pooling model
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
Details:
	•	Projection MLP (444→128→64)
	•	BiGRU
	•	Additive attention pooling (with mask)
	•	Classifier head (→4 logits)

"""

"""
model.py

BiGRU + Additive Attention Pooling for student-level classification.

Architecture (per your spec):
- Segment projection MLP: 444 -> 128 -> 64, dropout=0.3
- BiGRU hidden size: 64 (bidirectional => output dim 128)
- Additive attention size: 64
- Classifier head: 128 -> 64 -> 4

Input:
- x: FloatTensor (B, T, 444) padded
- mask: BoolTensor (B, T) where True indicates valid time steps

Output:
- logits: FloatTensor (B, 4)
- (optional) attn_weights: FloatTensor (B, T) attention distribution over segments
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# -----------------------------
# Utilities
# -----------------------------
def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    logits: (..., T)
    mask:   (..., T) bool
    returns softmax over dim with masked positions = 0.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # Put -inf on masked positions so softmax -> 0 there
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, neg_inf)
    attn = F.softmax(masked_logits, dim=dim)

    # In case an entire row is masked (shouldn't happen if lengths>=1), replace NaNs with 0
    attn = torch.nan_to_num(attn, nan=0.0)
    return attn


# -----------------------------
# Model config
# -----------------------------
@dataclass
class ModelConfig:
    in_dim: int = 444
    proj_hidden: int = 128
    proj_out: int = 64
    dropout: float = 0.3
    gru_hidden: int = 64
    attn_hidden: int = 64
    num_classes: int = 4
    layer_norm: bool = True


# -----------------------------
# Additive attention pooling
# -----------------------------
class AdditiveAttentionPooling(nn.Module):
    """
    Additive attention pooling:
      e_t = v^T tanh(W h_t + b)
      alpha = softmax(e_t) with mask
      context = sum_t alpha_t * h_t

    h_t: (B, T, H)
    mask: (B, T)
    """

    def __init__(self, in_dim: int, attn_hidden: int):
        super().__init__()
        self.W = nn.Linear(in_dim, attn_hidden, bias=True)
        self.v = nn.Linear(attn_hidden, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B,T,H)
        # score: (B,T)
        u = torch.tanh(self.W(h))               # (B,T,A)
        score = self.v(u).squeeze(-1)           # (B,T)

        alpha = masked_softmax(score, mask, dim=-1)  # (B,T)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, H)

        return context, alpha


# -----------------------------
# Main model
# -----------------------------
class StudentBiGRUAttnClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg

        # Segment projection MLP: 444 -> 128 -> 64
        layers = [
            nn.Linear(cfg.in_dim, cfg.proj_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.proj_hidden, cfg.proj_out),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        ]
        self.proj = nn.Sequential(*layers)

        self.proj_ln = nn.LayerNorm(cfg.proj_out) if cfg.layer_norm else nn.Identity()

        # BiGRU: input=64, hidden=64, bidirectional => output=128
        self.bigru = nn.GRU(
            input_size=cfg.proj_out,
            hidden_size=cfg.gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        rnn_out_dim = cfg.gru_hidden * 2

        # Additive attention pooling over BiGRU outputs
        self.attn = AdditiveAttentionPooling(in_dim=rnn_out_dim, attn_hidden=cfg.attn_hidden)

        # Classifier head: 128 -> 64 -> 4
        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.num_classes),
        )

    @torch.no_grad()
    def infer_lengths(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Convert (B,T) bool mask to lengths (B,) int64.
        Assumes valid steps are True and padding is False.
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        lengths = mask.sum(dim=1).to(torch.long)
        # clamp to at least 1 to keep pack_padded_sequence happy
        lengths = torch.clamp(lengths, min=1)
        return lengths

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x:    (B, T, 444) padded
        mask: (B, T) bool, True = valid

        returns:
          logits: (B, 4)
          attn_weights (optional): (B, T)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B,T,F), got {tuple(x.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"mask must be (B,T), got {tuple(mask.shape)}")
        if x.shape[0] != mask.shape[0] or x.shape[1] != mask.shape[1]:
            raise ValueError(f"shape mismatch: x={tuple(x.shape)} mask={tuple(mask.shape)}")

        B, T, Fdim = x.shape
        if Fdim != self.cfg.in_dim:
            raise ValueError(f"feature dim mismatch: got {Fdim}, expected {self.cfg.in_dim}")

        # Projection on each segment vector
        z = self.proj(x)          # (B,T,64)
        z = self.proj_ln(z)       # (B,T,64)

        lengths = self.infer_lengths(mask)  # (B,)

        # Pack padded sequence for efficient GRU
        # pack requires CPU lengths
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bigru(packed)
        h, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # (B,T,128)

        # Attention pooling with mask
        context, alpha = self.attn(h, mask)  # context: (B,128), alpha: (B,T)

        # Class logits
        logits = self.head(context)  # (B,4)

        if return_attn:
            return logits, alpha
        return logits, None


# -----------------------------
# Quick self-test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = ModelConfig()
    model = StudentBiGRUAttnClassifier(cfg)

    B, T = 4, 7
    x = torch.randn(B, T, cfg.in_dim)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, :7] = True
    mask[1, :5] = True
    mask[2, :2] = True
    mask[3, :1] = True  # minimal length 1

    logits, attn = model(x, mask, return_attn=True)
    print("logits:", logits.shape)  # (B,4)
    print("attn:", attn.shape)      # (B,T)
    print("attn row sums:", attn.sum(dim=1))
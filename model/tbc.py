# =============================================================================
# TransformerForBC: A simple transformer for Binary Classification
# -----------------------------------------------------------------------------
# Summary: Implements a simple transformer module used for binary
#          classification on synthetic data.
# Author: Q.WU
# =============================================================================

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


class TransformerForBC(nn.Module):
    def __init__(self, d=32, L=10, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_enc = PositionalEncoding(d, max_len=L + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=4 * d, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        # return only [CLS] token's logits
        return self.fc(x[:, 0, :])

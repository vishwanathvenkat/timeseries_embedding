from torch import nn
from utils import PositionalEncoding
import math
from functools import wraps


"""
Notes:
1. We dont need embedding since its a time series and we dont have any semantic meaning
2. 
"""


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        # num_tokens,
        d_model,
        nhead,
        num_encoder_layers,
        dropout = 0.1
    ):
        super().__init__()

        self.dim_model = d_model
        # LAYERS
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout), num_encoder_layers
        )
        self.linear = nn.Linear(d_model, d_model)  # Project to same dimensionality

    def forward(
        self,
        data):
        encoded = self.encoder(data)
        return self.linear(encoded)
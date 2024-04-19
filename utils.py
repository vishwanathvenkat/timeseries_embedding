import torch
import math
from torch import nn
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def read_data(path):
    data = np.load(path)
    return data


def convert_data_to_tensor(data):
    return torch.tensor(data)


def convert_tensor_to_dataset(data):
    return torch.utils.data.TensorDataset(data)


def convert_dataset_to_dataloader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle)


def split_data(data):
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, val_data


def create_dataloader(path):
    # Read the data
    data = read_data(path)
    data_tensor = convert_data_to_tensor(data)

    # Split the data
    train_data, val_data = train_test_split(data_tensor, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(train_data)
    val_dataset = torch.utils.data.TensorDataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_dataloader, val_dataloader


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

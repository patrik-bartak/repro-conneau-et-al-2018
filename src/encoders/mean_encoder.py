import torch
import torch.nn as nn


class MeanEncoder(nn.Module):
    def __init__(self, **_):
        """
        Initialize an encoder that takes the mean of the input token vectors.
        """
        super(MeanEncoder, self).__init__()

    def forward(self, embs: torch.Tensor, _) -> torch.Tensor:
        emb = torch.mean(embs, dim=1)
        return emb

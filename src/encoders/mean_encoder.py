import torch
import torch.nn as nn


class MeanEncoder(nn.Module):
    def __init__(self, **_):
        super(MeanEncoder, self).__init__()

    def forward(self, embs, _):
        emb = torch.mean(embs, dim=1)
        return emb

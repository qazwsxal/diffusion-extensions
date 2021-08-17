import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResLayer(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x+self.layer(x)

class Siren(nn.Module):
    """
    Encodes spatial/temporal dimensions as SIREN representation
    https://arxiv.org/abs/2006.09661

    in_channels -
    out_channels -
    scale - Scaling factor for weight initialisation, use 30 for first layer if inputs are +-1
    post_scale - Scaling factor post-siren, helps a bit when running the output through ResNets
    optimise - optimize parameters - set to False
    """

    def __init__(self, in_channels, out_channels, scale=1, optimize=True, post_scale=True):
        super().__init__()
        self.positional = nn.Linear(in_features=in_channels, out_features=out_channels)

        # See "3.2 Distribution of activations, frequencies, and a principled initialization scheme" in paper.
        torch.nn.init.uniform_(self.positional.weight, -(6 / in_channels) ** 0.5, (6 / in_channels) ** 0.5)
        self.positional.weight.data *= scale

        # bias terms should cover +-pi
        torch.nn.init.uniform_(self.positional.bias, -3.14159, 3.14159)
        if post_scale:
            self.post_scale = nn.Linear(out_channels, out_channels)
        else:
            self.post_scale = None
        # Freeze parameters if we don't want to optimise
        for param in self.parameters():
            param.requires_grad = optimize

    def forward(self, x):
        res = torch.sin(self.positional(x))
        if self.post_scale:
            return self.post_scale(res)
        else:
            return res
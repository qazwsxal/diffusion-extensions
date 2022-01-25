import math
from typing import Tuple

import torch
from se3_transformer_pytorch.se3_transformer_pytorch import LinearSE3, Fiber, NormSE3
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from prot_util import RES_COUNT
from util import ProtData, AffineGrad, euler_to_rmat


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
        return x + self.layer(x)


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


class PointCloudProj(nn.Module):
    def __init__(self, data, so3=True):
        super().__init__()
        self.data = data
        self.so3 = so3

    def forward(self, x):
        # The transpose operation here here is due to the shape of self.data.
        # (A^T)^T = A
        # (AB)^T = B^T A^T
        # So for rotation R and data D:
        # (RD^T)^T = (D^T)^T R^T = D R^T
        if self.so3:
            R_T = x.transpose(-1, -2)
        else:
            R_T = euler_to_rmat(*torch.unbind(x, -1)).transpose(-1, -2)
        return self.data @ R_T


class PoolRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask=None):
        if mask == None:
            mask = torch.ones((*x.shape[:-1], 1), dtype=torch.bool).to(x.device)
        weight = (self.pool(x) * mask)
        w_sum = weight.sum(dim=-2, keepdim=True).clamp(min=1e-6)
        val = self.lin(x)
        out = (val * weight).sum(dim=-2, keepdim=True) / w_sum
        return out[..., 0, :]


class PoolSE3(nn.Module):
    def __init__(self, fiber):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(fiber["0"], 1),
            nn.Sigmoid(),
        )
        self.lin = FFSE3(fiber, fiber)

    def forward(self, x, mask):
        weight = (self.pool(x["0"][..., 0]) * mask[..., None]).unsqueeze(-1)
        w_sum = weight.sum(dim=-3, keepdim=True).clamp(min=1e-6)
        val = self.lin(x)
        out = {k: (v * weight).sum(dim=-3, keepdim=True) / w_sum for k, v in val.items()}
        return out


class FFSE3(nn.Module):
    def __init__(
            self,
            fiber_in,
            fiber_out,
            gated_scale=False,
            mult=4,
    ):
        super().__init__()
        self.fiber = fiber_in
        fiber_hidden = Fiber(list(map(lambda t: (t[0], t[1] * mult), fiber_in)))

        self.project_in = LinearSE3(fiber_in, fiber_hidden)
        self.nonlin = NormSE3(fiber_hidden, gated_scale=gated_scale)
        self.project_out = LinearSE3(fiber_hidden, fiber_out)

    def forward(self, features):
        outputs = self.project_in(features)
        outputs = self.nonlin(outputs)
        outputs = self.project_out(outputs)
        return outputs


class PlaneNet(nn.Module):
    def __init__(self, dim=512, heads=4, layers=4):
        super().__init__()
        d_out = 3
        enc_layer = nn.TransformerEncoderLayer(dim, heads)
        self.position_siren = Siren(in_channels=3, out_channels=dim // 2, scale=30)
        self.time_embedding = SinusoidalPosEmb(dim // 2)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))

        self.out_net = nn.Sequential(PoolRN(dim),
                                     nn.Linear(dim, 3),
                                     )

    def forward(self, x, t):
        batch = x.shape[0]
        x_emb = self.position_siren(x)
        t_emb = self.time_embedding(t)
        t_in = torch.cat((x_emb, t_emb[:, None, :].expand(x_emb.shape)), dim=2)
        # Transpose batch and sequence dimension
        # Because we're using a version of PT that doesn't support
        # Batch first.
        encoding = self.encoder(t_in.transpose(0, 1)).transpose(0, 1)
        out = self.out_net(encoding)
        return out[..., 0, :]  # Drop sequence dimension


class ProtNet(nn.Module):
    def __init__(self, dim=64, heads=4, t_depth=4,
                 c_depth=3, se3=True):
        super().__init__()
        time_dim = dim // 4
        pos_dim = dim // 3
        ang_dim = dim // 3
        res_dim = dim - (time_dim + pos_dim + ang_dim)
        self.se3 = se3
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.pos_emb = Siren(3, pos_dim)
        self.ang_emb = Siren(9, ang_dim)
        # 1-d conv block, res_count -> dim -> dim -> dim ... dim -> res_dim
        # intermediate layers (dim -> dim) are residual (linear + SiLU) defined by list comprehension
        self.res_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=RES_COUNT,
                out_channels=dim,
                kernel_size=(3,),
                padding=(1,),
                stride=(1,)
            ),
            nn.SiLU(inplace=True),
            *[ResLayer(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=dim,
                        out_channels=dim,
                        kernel_size=(3,),
                        padding=(1,),
                        stride=(1,)
                    ),
                    nn.SiLU(inplace=True),
                )
            )
                for _ in range(c_depth - 2)
            ],
            nn.Conv1d(
                in_channels=dim,
                out_channels=res_dim,
                kernel_size=(3,),
                padding=(1,),
                stride=(1,)
            ),
        )
        enc_layer = nn.TransformerEncoderLayer(dim, heads)
        encoder_norm = nn.LayerNorm(dim, eps=1e-5)
        self.trans = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=t_depth,
            norm=encoder_norm
        )

        self.pool = PoolRN(dim)
        self.last = nn.Sequential(nn.SiLU(),
                                  nn.Linear(dim, 6),
                                  )

    def forward(self, x: Tuple[Tuple[ProtData, ProtData]], t):

        ang = pad_sequence([torch.cat((r.angles, l.angles), dim=0) for r, l in x], batch_first=True)
        ang_flat = ang.flatten(-2, -1)
        pos = pad_sequence([torch.cat((r.positions, l.positions), dim=0) for r, l in x], batch_first=True)
        res_embed = pad_sequence([torch.cat((
            self.res_conv(r.residues[None].transpose(-1, -2)).transpose(-1, -2)[0],
            self.res_conv(l.residues[None].transpose(-1, -2)).transpose(-1, -2)[0],
        ), dim=0) for r, l in x], batch_first=True)
        time_embed = self.time_emb(t)
        pos_embed = self.pos_emb(pos)
        ang_embed = self.ang_emb(ang_flat)
        # If there's no onehot'd residue, then it's a pad value (all 0's).
        # Need True to mask out
        msk = pos.any(dim=-1)
        seq_len = msk.shape[1]
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)
        t_in = torch.cat((time_embed, res_embed, pos_embed, ang_embed), dim=-1)
        t_out = self.trans(t_in.transpose(0, 1), src_key_padding_mask=msk.logical_not()).transpose(0, 1)

        pool_out = self.pool(t_out)
        last_out = self.last(pool_out)
        if self.se3:
            out = AffineGrad(rot_g=last_out[..., :3], shift_g=last_out[..., 3:])
        else:
            out = last_out
        return out

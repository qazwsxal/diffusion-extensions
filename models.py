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
        weight = (self.pool(x) * mask[..., None])
        w_sum = weight.sum(dim=-2, keepdim=True).clamp(min=1e-6)
        val = self.lin(x)
        out = (val * weight).sum(dim=-2, keepdim=True) / w_sum
        return out[..., 0, :]


class PoolPos(nn.Module):
    def __init__(self, dim_pool):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(dim_pool, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, pos, mask=None):
        if mask == None:
            mask = torch.ones((*x.shape[:-1], 1), dtype=torch.bool).to(x.device)
        weight = (self.pool(x) * mask[..., None])
        w_sum = weight.sum(dim=-2, keepdim=True).clamp(min=1e-6)
        out = (pos * weight).sum(dim=-2, keepdim=True) / w_sum
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


class TransformerEnc2(nn.Module):
    def __init__(self, dim=512, heads=4, layers=4):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(dim, heads)
        encoder_norm = nn.LayerNorm(dim, eps=1e-5)
        self.encoder = nn.TransformerEncoder(enc_layer, layers, norm=encoder_norm)

    def forward(self, x, src_key_padding_mask=None):
        # Transpose batch and sequence dimension
        # Because we're using a version of PT that doesn't support
        # Batch first.
        encoding = self.encoder(x.transpose(0, 1), src_key_padding_mask=src_key_padding_mask).transpose(0, 1)
        return encoding  # Drop sequence dimension


class PlaneNet(nn.Module):
    def __init__(self, dim=512, heads=4, layers=4):
        super().__init__()
        dim_out = 3

        enc_layer = nn.TransformerEncoderLayer(dim, heads)

        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.position_siren = Siren(in_channels=3, out_channels=dim // 2, scale=30)
        self.time_embedding = SinusoidalPosEmb(dim // 2)

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
        time_dim = dim
        pos_dim = dim // 2
        ang_dim = dim // 4
        res_dim = dim - (pos_dim + ang_dim)
        self.se3 = se3
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.pos_emb = Siren(3, pos_dim, scale=0.1)
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
        self.lig_tf = TransformerEnc2(dim=dim, layers=t_depth, heads=heads)
        self.lig_emb_pool = PoolRN(dim)
        self.lig_pos_pool = PoolPos(dim)
        self.rec_tf = TransformerEnc2(dim=dim, layers=t_depth, heads=heads)
        self.rec_emb_pool = PoolRN(dim)
        self.rec_pos_pool = PoolPos(dim)

        self.last = nn.Sequential(nn.Sequential(nn.Linear(3 * dim + 6, dim),
                                                nn.SiLU(inplace=True),
                                                ),
                                  *[ResLayer(nn.Sequential(nn.Linear(dim, dim),
                                                           nn.SiLU(inplace=True),
                                                           ))
                                    for _ in range(3)],
                                  nn.Linear(dim, 6),
                                  )

    def forward(self, x: Tuple[Tuple[ProtData, ProtData]], t):
        time_embed = self.time_emb(t)
        r_ang = pad_sequence([r.angles for r, _ in x], batch_first=True)
        r_ang_flat = r_ang.flatten(-2, -1)
        r_ang_embed = self.ang_emb(r_ang_flat)
        r_pos = pad_sequence([r.positions for r, _ in x], batch_first=True)
        r_pos_embed = self.pos_emb(r_pos)
        r_res_embed = pad_sequence([self.res_conv(r.residues[None].transpose(-1, -2)).transpose(-1, -2)[0]
                                    for r, _ in x], batch_first=True)

        # If there's no onehot'd residue, then it's a pad value (all 0's).
        # Need True to mask out
        r_msk = r_pos.any(dim=-1)
        r_seq_len = r_msk.shape[1]
        r_t_in = torch.cat((r_res_embed, r_pos_embed, r_ang_embed), dim=-1)
        r_t_out = self.rec_tf(r_t_in, src_key_padding_mask=r_msk.logical_not())

        r_pool_out = self.rec_emb_pool(r_t_out, r_msk)
        r_pos_out = self.rec_pos_pool(r_t_out, r_pos, r_msk)

        l_ang = pad_sequence([l.angles for _, l in x], batch_first=True)
        l_ang_flat = l_ang.flatten(-2, -1)
        l_ang_embed = self.ang_emb(l_ang_flat)
        l_pos = pad_sequence([l.positions for _, l in x], batch_first=True)
        l_pos_embed = self.pos_emb(l_pos)
        l_res_embed = pad_sequence([self.res_conv(l.residues[None].transpose(-1, -2)).transpose(-1, -2)[0]
                                    for _, l in x], batch_first=True)

        # If there's no onehot'd residue, then it's a pad value (all 0's).
        # Need True to mask out
        l_msk = l_pos.any(dim=-1)
        l_seq_len = l_msk.shape[1]
        l_t_in = torch.cat((l_res_embed, l_pos_embed, l_ang_embed), dim=-1)
        l_t_out = self.rec_tf(l_t_in, src_key_padding_mask=l_msk.logical_not())

        l_pool_out = self.lig_emb_pool(l_t_out, l_msk)
        l_pos_out = self.lig_pos_pool(l_t_out, l_pos, l_msk)

        pool = torch.cat((time_embed, r_pool_out, r_pos_out, l_pool_out, l_pos_out), dim=-1)
        last_out = self.last(pool)
        if self.se3:
            out = AffineGrad(rot_g=last_out[..., :3], shift_g=last_out[..., 3:])
        else:
            out = last_out
        return out

import math
from typing import Tuple

import torch
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.se3_transformer_pytorch import LinearSE3, Fiber
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from prot_util import RES_COUNT
from util import ProtData, AffineGrad, masked_mean


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
    def __init__(self, data):
        super().__init__()
        self.data = data

    def forward(self, x):
        return self.data @ x.transpose(-1, -2)


class SE3Pool(nn.Module):
    def __init__(self, fiber):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Linear(fiber["0"], 1),
            nn.Sigmoid(),
            )
        self.lin = LinearSE3(fiber, fiber)

    def forward(self, x, mask):
        weight = (self.pool(x["0"][..., 0]) * mask[..., None]).unsqueeze(-1)
        w_sum = weight.sum(dim=-3, keepdim=True)
        val = self.lin(x)
        out = {k: (v * weight).sum(dim=-3, keepdim=True) / w_sum for k, v in val.items()}
        return out


class ProtNet(nn.Module):
    def __init__(self, dim=64, heads=4, t_depth=4, dim_head=16, num_degrees=3, num_neighbours=10,
                 c_depth=3, pool_depth=4):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(dim // 2)
        self.se3trans = SE3Transformer(
            dim=dim,
            heads=heads,
            depth=t_depth,
            dim_head=dim_head,
            input_degrees=2,
            num_degrees=num_degrees,
            output_degrees=num_degrees,
            # this must be set to true, in which case it will assert that you pass in the adjacency matrix
            attend_sparse_neighbors=True,
            # if you set this to 0, it will only consider the connected neighbors as defined by the adjacency matrix.
            # but if you set a value greater than 0, it will continue to fetch the closest points up to this many,
            # excluding the ones already specified by the adjacency matrix
            num_neighbors=num_neighbours,
            # GLA removed in newer package versions?
            # # We're interested in a single, global transformation,
            # # so take a global linear attention
            # # as otherwise we won't actually get global information flow
            # # due to the nearest-neighbour restriction present in SE3 transformers,
            # global_linear_attn_every=1,
            reversible=True,
            )
        self.pooltrans = SE3Transformer(
            dim=dim,
            heads=heads,
            depth=pool_depth,
            dim_head=dim_head,
            input_degrees=num_degrees,
            num_degrees=num_degrees,
            output_degrees=2,
            num_neighbors=num_neighbours,
            reversible=True,
            )
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
                for _ in range(c_depth - 1)
                ],
            )
        # We need to generate an equal number of type-0 and type-1 features
        # So use an explict SE3-invariant linear transform to project up.
        self.vec_proj = LinearSE3(Fiber({"1": 3}),
                                  Fiber({"1": dim}),
                                  )

        self.downsample = LinearSE3(Fiber({str(x): dim for x in range(num_degrees)}),
                                    Fiber({**{"0": dim // 2},
                                           **{str(x): dim for x in range(1,num_degrees)}}),
                                    )

        self.final_se3 = LinearSE3(Fiber({"0": dim,
                                          "1": dim}),
                                   Fiber({"1": 2}),
                                   )

        self.pool = SE3Pool(Fiber({str(k): dim for k in range(num_degrees)}))

    def forward(self, x: Tuple[Tuple[ProtData, ProtData]], t):
        device = x[0][0][0].device
        receptors, ligands = [x for x in zip(*x)]

        r_ang = pad_sequence([r.angles for r in receptors], batch_first=True)
        r_pos = pad_sequence([r.positions for r in receptors], batch_first=True)
        r_res = pad_sequence([r.residues for r in receptors], batch_first=True)
        r_msk = pad_sequence([torch.ones(len(r.angles)) for r in receptors], batch_first=True).to(bool).to(device)
        r_adj = torch.diag_embed(r_msk[..., :-1], offset=1)
        r_adj = torch.logical_or(r_adj, r_adj.transpose(-1, -2))

        r_emb = self.res_conv(r_res.transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        r_fea = self.vec_proj({"1": r_ang})
        r_fea["0"] = r_emb
        r_out = self.se3trans(r_fea, r_pos, r_msk, r_adj, return_pooled=False)
        r_out["0"] = r_out["0"].unsqueeze(-1)

        r_pool = self.pool(r_out, r_msk)

        l_ang = pad_sequence([l.angles for l in ligands], batch_first=True)
        l_pos = pad_sequence([l.positions for l in ligands], batch_first=True)
        l_res = pad_sequence([l.residues for l in ligands], batch_first=True)
        l_msk = pad_sequence([torch.ones(len(l.angles)) for l in ligands], batch_first=True).to(bool).to(device)
        l_adj = torch.diag_embed(l_msk[..., :-1], offset=1)
        l_adj = torch.logical_or(l_adj, l_adj.transpose(-1, -2))

        l_emb = self.res_conv(l_res.transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        l_fea = self.vec_proj({"1": l_ang})
        l_fea["0"] = l_emb
        l_out = self.se3trans(l_fea, l_pos, l_msk, l_adj, return_pooled=False)

        l_out["0"] = l_out["0"].unsqueeze(-1)

        l_pool = self.pool(l_out, l_msk)

        pool = {k: torch.cat((r_pool[k], l_pool[k]), dim=1) for k in r_pool.keys()}
        pool_ds = self.downsample(pool)
        pos = torch.stack((masked_mean(r_pos, r_msk, dim=1),
                           masked_mean(l_pos, l_msk, dim=1)),
                          dim=1)
        time_emb = self.time_emb(t)[..., None, :, None].expand_as(pool_ds["0"])
        pool_ds["0"] = torch.cat((pool_ds["0"], time_emb), dim=-2)
        p_out = self.pooltrans(pool_ds, pos)
        p_out["0"] = p_out["0"].unsqueeze(-1)
        out = self.final_se3(p_out)
        # shape = [b, (r/l), feat, dim], look at ligand output.
        aff_out = AffineGrad(rot_g=out["1"][:, 1, 0], shift_g=out["1"][:, 1, 1])
        return aff_out
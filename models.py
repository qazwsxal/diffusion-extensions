import math
from typing import Tuple

import torch
from einops import rearrange
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.se3_transformer_pytorch import LinearSE3, Fiber
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from prot_util import RES_COUNT
from util import ProtData, AffineGrad


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


class SingleQAttentionSE3(nn.Module):
    # Dodgy Attention block that has only a single query token.
    # allows us to do SE3 attention without the nearest-neighbour bottleneck
    # Alternative to taking the mean that allows us to select relevant tokens,
    # O(n) complexity due to attention matrix being 1*n
    # "Dodgy" because we need to preserve SE3 invariance,
    # A learned query token with type-(1,2,...) features would not be invariant to rotation.
    # so we only calculate the attention based on type-0 features.

    def __init__(self, fiber, dim_head, heads):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, heads, 1, dim_head))
        self.W_k = nn.Parameter(torch.randn((heads, fiber["0"], dim_head)) / fiber["0"] ** 0.5)
        self.dim_head = dim_head
        head_fiber = Fiber({"0": dim_head,
                            "1": dim_head,
                            })

        self.to_v = nn.ModuleList([LinearSE3(fiber, head_fiber) for _ in range(heads)])
        headmix_fiber = Fiber({"0": dim_head * heads,
                               "1": dim_head * heads,
                               })
        self.lin_out = LinearSE3(headmix_fiber, fiber)

    def forward(self, x):
        # TODO finish implementation and fix dimensionality of fibers
        k = torch.einsum("b t d i, h d e -> b h t e", x["0"], self.W_k)
        attn = torch.einsum("...i, ...i -> ...", k, self.q) / (self.dim_head ** 0.5)
        attn = attn.softmax(-1)
        v = [v_lin(x) for v_lin in self.to_v]

        v_head = {k: torch.stack([f[k] for f in v], dim=1) for k in v[0].keys()}

        attn_head_out = {dk: torch.einsum("b h t, b h t d i -> b h d i", attn, dv) for dk, dv in v_head.items()}
        attn_out = {dk: rearrange(dv, 'b h d m -> b () (h d) m') for dk, dv in attn_head_out.items()}

        out = self.lin_out(attn_out)
        return out


class ProtNet(nn.Module):
    # TODO global attention
    def __init__(self, dim=64, heads=4, t_depth=4, dim_head=16, num_degrees=3, num_neighbours=10,
                 c_depth=3):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(dim // 2)
        self.se3trans = SE3Transformer(
            dim=dim,
            heads=heads,
            depth=t_depth,
            dim_head=dim_head,
            input_degrees=2,
            num_degrees=num_degrees,
            output_degrees=2,
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
            )
        self.res_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=RES_COUNT + 1,
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
            nn.Conv1d(
                in_channels=dim,
                out_channels=dim // 2,
                kernel_size=(3,),
                padding=(1,),
                stride=(1,)
                ),
            )
        # We need to generate an equal number of type-0 and type-1 features
        # So use an explict SE3-invariant linear transform to project up.
        self.vec_proj = LinearSE3(Fiber({"1": 3}),
                                  Fiber({"1": dim}),
                                  )

        self.last_attn = SingleQAttentionSE3(Fiber({"0": dim, "1": dim}), dim_head=dim_head, heads=heads)
        self.downsample = LinearSE3(Fiber({"0": dim,
                                           "1": dim}),
                                    Fiber({"1": 2}),
                                    )

    def forward(self, x: Tuple[Tuple[ProtData, ProtData]], t):
        device = x[0][0][0].device
        masksize = max(len(prots[0].positions) + len(prots[1].positions) for prots in x)
        mask = torch.ones(len(x), masksize, dtype=torch.bool).to(device)
        adjv = torch.ones(len(x), masksize - 1, dtype=torch.bool).to(device)

        anglist = [torch.cat((rec.angles, lig.angles), dim=0) for rec, lig in x]
        poslist = [torch.cat((rec.positions, lig.positions), dim=0) for rec, lig in x]
        reslist = [torch.cat((rec.residues, lig.residues), dim=0) for rec, lig in x]
        ligmask = [torch.cat((torch.zeros(len(rec.residues), 1),
                              torch.ones(len(lig.residues), 1)),
                             dim=0).to(device)
                   for rec, lig in x]
        reslist = [torch.cat((res, lm), dim=-1) for res, lm in zip(reslist, ligmask)]

        angles = pad_sequence(anglist, batch_first=True)
        positions = pad_sequence(poslist, batch_first=True)
        residues = pad_sequence(reslist, batch_first=True)

        resd_emb = self.res_conv(residues.transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        time_emb = self.time_emb(t)[..., None, :, None].expand_as(resd_emb)
        zero_emb = torch.cat((resd_emb, time_emb), dim=-2)

        features = self.vec_proj({"1": angles})

        features["0"] = zero_emb

        for i, (residue, ligand) in enumerate(x):
            mask[i, (len(residue[0]) + len(residue[1])):] = False
            adjv[i, len(residue[0])] = False
        adj_mat = torch.diag_embed(adjv, offset=1).to(device)
        adj_mat = adj_mat + adj_mat.transpose(-1, -2)
        adj_mat = adj_mat.to(torch.bool)

        t_out = self.se3trans(features, positions, mask, adj_mat, return_pooled=False)
        # annoying squeeze of last dimension built into transformer model.
        t_out["0"] = t_out["0"].unsqueeze(-1)
        attn_out = self.last_attn(t_out)
        out = self.downsample(attn_out)
        return AffineGrad(rot_g=out["1"][:,0, 0], shift_g=out["1"][:,0, 1])
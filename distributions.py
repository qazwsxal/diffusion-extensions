from math import pi

import numpy as np
import torch
from torch.distributions import Distribution, constraints, Normal

from util import *


class IsotropicGaussianSO3(Distribution):
    arg_constraints = {'eps': constraints.positive}

    def __init__(self, eps: torch.Tensor, mean: torch.Tensor = torch.eye(3)):
        self.eps = eps
        self._mean = mean.to(self.eps)
        self._mean_inv = self._mean.transpose(-1, -2)  # orthonormal so inverse = Transpose
        pdf_sample_locs = pi * torch.linspace(0, 1.0, 1000) ** 3.0  # Pack more samples near 0
        pdf_sample_locs = pdf_sample_locs.to(self.eps)
        # As we're sampling using axis-angle form
        # and need to account for the change in density
        # Scale by 1-cos(t)/pi for sampling
        with torch.no_grad():
            pdf_sample_vals = self._eps_ft(pdf_sample_locs) * ((1 - pdf_sample_locs.cos()) / pi)
        # Set to 0.0, otherwise there's a divide by 0 here
        pdf_sample_vals[(pdf_sample_locs == 0).expand_as(pdf_sample_vals)] = 0.0
        # There's a weird oscillation introduced with small eps, but only for large theta.
        # manually fix by setting every value from the first one evaluated as negative onwards to 0.0
        is_neg = pdf_sample_vals < 0.0
        any_neg = is_neg.any(dim=-1)
        first_neg = torch.argmax(is_neg.long(), dim=-1)
        for i, (fn, an) in enumerate(zip(first_neg,any_neg)):
            if an.item():
                pdf_sample_vals[i,fn:] = 0.0
        # Trapezoidal intergration
        pdf_val_sums = pdf_sample_vals[..., :-1] + pdf_sample_vals[..., 1:]
        pdf_loc_diffs = torch.diff(pdf_sample_locs, dim=0)
        self.trap = (pdf_loc_diffs * pdf_val_sums / 2).cumsum(dim=-1)
        self.trap /= self.trap[:, -1, None].clone()
        self.trap_loc = pdf_sample_locs[1:]
        super().__init__()


    def sample(self, sample_shape=torch.Size()):
        # Consider axis-angle form.
        axes = torch.randn((*sample_shape, *self.mean.shape[:-2], 3)).to(self.eps)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        # Inverse transform sampling based on numerical approximation of CDF
        unif = torch.rand((*sample_shape, *self.mean.shape[:-2]), device=self.trap.device)
        idx_1 = (self.trap < unif[..., None]).sum(dim=-1)
        idx_0 = idx_1 - 1

        trap_start = self.trap[range(self.trap.shape[0]), idx_0]
        trap_end = self.trap[range(self.trap.shape[0]), idx_1]
        weight = ((unif - trap_start) / (trap_end - trap_start))
        angle_start = self.trap_loc[idx_0]
        angle_end = self.trap_loc[idx_1]
        angles = torch.lerp(angle_start, angle_end, weight)[..., None]
        out = self._mean @ aa_to_rmat(axes, angles)
        return out

    def _eps_ft_inner(self, l, t: torch.Tensor) -> torch.Tensor:
        lt_sin = torch.sin((l + 0.5) * t) / torch.sin(t / 2)
        lt_sin[t[..., 0] == 0.0] = ((l + 0.5) / 0.5)
        newdims = (1,) * len(lt_sin.shape)
        eps_exp = self.eps.view(-1, *newdims)
        out = (2 * l + 1) * torch.exp(-l * (l + 1) * (eps_exp ** 2)) * lt_sin
        return out

    def _eps_ft(self, t: torch.Tensor) -> torch.Tensor:
        maxdims = max(len(self.eps.shape), len(t.shape))
        # This is an infinite sum, approximate with 5/eps**2 values
        l_count = round(min(max(1e3,(5 / self.eps.min() ** 2).item()), 1e7))
        if l_count >= 1e5:
            chunk_size = int(1e5) # Should tune this
            print("small eps! Using chunked l calcs", self.eps.min())
            vals = torch.zeros_like(t[None])
            for l_min in range(0, l_count, chunk_size):
                l_max = l_min + chunk_size
                l = torch.arange(l_min, l_max).reshape((*([1] * maxdims), -1)).to(self.eps)
                inner = self._eps_ft_inner(l, t[..., None])
                inner_vals = inner.sum(dim=-1)
                vals += inner_vals
        else:
            l = torch.arange(l_count).reshape((*([1] * maxdims), -1)).to(self.eps)
            inner = self._eps_ft_inner(l, t[..., None])
            vals = inner.sum(dim=-1)
        return vals

    def log_prob(self, rotations):
        _, angles = rmat_to_aa(rotations)
        probs = self._eps_ft(angles)
        return probs.log()

    @property
    def mean(self):
        return self._mean


class IGSO3xR3(Distribution):
    arg_constraints = {'eps': constraints.positive}

    def __init__(self, eps: torch.Tensor, mean: AffineT = None, shift_scale=1.0):
        self.eps = eps
        if mean == None:
            rot = torch.eye(3).unsqueeze(0)
            shift = torch.zeros(*eps.shape, 3).to(eps)#
            mean = AffineT(shift=shift, rot=rot)
        self._mean = mean.to(eps)
        self.igso3 = IsotropicGaussianSO3(eps=eps*rot_scale, mean=self._mean.rot)
        self.r3 = Normal(loc=self._mean.shift, scale=eps[...,None]*shift_scale)
        super().__init__()


    def sample(self, sample_shape=torch.Size()):
        rot = self.igso3.sample(sample_shape)
        shift = self.r3.sample(sample_shape)
        return AffineT(rot, shift)

    def log_prob(self, value):
        rot_prob = self.igso3.log_prob(value.rot)
        shift_prob = self.r3.log_prob(value.shift)
        return rot_prob + shift_prob

    @property
    def mean(self):
        return self._mean


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    epsilons = torch.linspace(0.01, 1.0, 10).to(device)
    dist = IsotropicGaussianSO3(epsilons)
    rot = dist.rsample()
    rot.requires_grad = True
    distance = rmat_dist(rot, torch.eye(3)[None])
    distance.sum().backward()
    rotations = dist.sample((1000,))
    for i, s in enumerate(epsilons):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(f"Epsilon = {s.item()}")
        ax.scatter(*rotations[:, i, 0, :].T)
        ax.scatter(*rotations[:, i, 1, :].T)
        ax.scatter(*rotations[:, i, 2, :].T)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        plt.show()

    axis = torch.randn((3,))
    axis = (axis / axis.norm(dim=-1, p=2, keepdim=True)).repeat(100, 1)
    axis.requires_grad = True
    angle = torch.linspace(0.001, pi / 2, steps=100).unsqueeze(-1)
    angle.requires_grad = True
    rmats = aa_to_rmat(axis, angle)
    dist2 = IsotropicGaussianSO3(torch.tensor(0.1))
    l_probs = dist2.log_prob(rmats)
    grads = torch.autograd.grad(l_probs.sum(), rmats, retain_graph=True)
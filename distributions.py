from functools import reduce
from math import pi, log, exp
from operator import mul

import torch
from torch.distributions import Distribution, constraints, Normal

from rotations import aa_to_rmat

class WrappedNormal(Distribution):
    arg_constraints = {'loc': constraints.real,
                       'scale': constraints.positive}

    def __init__(self, scale, loc: torch.Tensor = torch.tensor(0.0)):
        super().__init__()
        self.loc = loc.to(scale)
        self.scale = scale
        self._norm = Normal(loc=self.loc, scale=scale)

    def rsample(self, sample_shape=torch.Size()):
        normvals = self._norm.rsample(sample_shape)
        val = torch.remainder(normvals + pi, 2 * pi) - pi
        return val

    def log_prob(self, value):
        # look at points up to 10-sigmas away and sum over them
        sig_range = torch.arange(0, self.scale * 10, 2 * pi)
        sig_range = torch.cat((sig_range, -sig_range), dim=0).reshape(-1, *([1] * len(value.shape))).to(value)
        expand_vals = sig_range + value[None, ...]
        norm_lps = self._norm.log_prob(expand_vals)
        # better perf/precision that doing it manually.
        lps = torch.logsumexp(norm_lps, dim=0)
        return lps


class IsotropicGaussianSO3(Distribution):
    arg_constraints = {'epsilon': constraints.positive}

    def __init__(self, epsilon: torch.Tensor, mean: torch.Tensor = torch.eye(3)):
        super().__init__()
        self.eps = epsilon
        self._mean = mean.to(epsilon)
        self._mean_inv = self._mean.permute(-1, -2)  # orthonormal so inverse = Transpose
        self._wn = WrappedNormal(3 * self.eps)
        # Rejection sampling scaling factor.

        self.rej_log_C = self._eps_f0().log().to(epsilon) - self._wn.log_prob(
            torch.tensor(0.0).to(epsilon))

    def rsample(self, sample_shape=torch.Size()):
        # Consider axis-angle form.
        axes = torch.randn((*sample_shape, 3)).to(self.eps)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        samples = torch.zeros(sample_shape).to(self.eps)
        goodsamples = torch.zeros(sample_shape, dtype=torch.bool)
        while not goodsamples.all():
            # This is really inefficient as it'll replace already good samples
            # but hey, it works
            wn_samples = self._wn.rsample(sample_shape)
            wn_probs = self._wn.log_prob(wn_samples)
            ig_probs = self._eps_ft(wn_samples)
            U = torch.rand(sample_shape).to(self.eps).log()
            keep = (U <= (ig_probs - (wn_probs + self.rej_log_C)))
            samples[keep] = wn_samples[keep]
            goodsamples = torch.logical_or(goodsamples, keep.cpu())

        return self._mean @ aa_to_rmat(axes, samples.sin(), samples.cos())

    def _eps_f0_inner(self, l):
        # limit as t -> 0
        lt_sin = 2 * l + 1
        return (2 * l + 1) * torch.exp(-l * (l + 1) * self.eps ** 2) * lt_sin

    def _eps_ft_inner(self, l, t: torch.Tensor) -> torch.Tensor:
        lt_sin = (torch.sin((l + 0.5) * t)) / torch.sin(t / 2)
        return (2 * l + 1) * torch.exp(-l * (l + 1) * self.eps ** 2) * lt_sin

    def _eps_f0(self) -> torch.Tensor:
        maxdims = len(self.eps.shape)
        # This is an infinite sum, approximate with 10/eps values
        l_count = min(torch.round(10 / self.eps).item(), 1e6)
        if l_count == 1e6:
            print("Very small epsilon!", self.eps)
        l = torch.arange(l_count).reshape((-1, *([1] * maxdims))).to(self.eps)
        inner = self._eps_f0_inner(l)
        return inner.sum(dim=0)

    def _eps_ft(self, t: torch.Tensor) -> torch.Tensor:
        maxdims = max(len(self.eps.shape), len(t.shape))
        # This is an infinite sum, approximate with 10/eps values
        l_count = min(torch.round(10 / self.eps).item(), 1e6)
        if l_count == 1e6:
            print("Very small epsilon!", self.eps)
        l = torch.arange(l_count).reshape((-1, *([1] * maxdims))).to(self.eps)
        inner = self._eps_ft_inner(l, t)

        return inner.sum(dim=0)

    @property
    def mean(self):
        return self._mean


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plot_util import update_line
    from matplotlib.animation import FuncAnimation

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    line1, = ax.plot([], [], [])


    def init():
        line1.set_data([], [])
        return line1,


    def animate(i):
        global new_empty
        update_line(line1, new_empty)
        new_empty = new_empty @ res[i].T
        return line1,


    # Define "axes" to plot with line
    empty = torch.tensor([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 1],
                          ], dtype=torch.float32).to(device)
    new_empty = empty.clone()
    # Create animation steps
    eps = torch.tensor(0.2).to(device)
    steps = 1000

    dist = IsotropicGaussianSO3(epsilon=eps)
    res = dist.rsample((steps,))

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=steps, interval=20, blit=True)

    anim.save(f'diffusion_{eps}.gif', writer='imagemagick')

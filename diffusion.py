import numpy as np
import torch


class DiffusionProcess(object):
    def __init__(self, steps: int, schedule: str, s=0.008, device=torch.device('cpu')):
        self.steps = steps
        if schedule == "cos" or schedule == "cosine":
            t = torch.linspace(0, 1, steps=steps+1, device=device)
            num = t + s
            denom = 1 + s
            alpha_bar = torch.cos((num / denom) * np.pi / 2)**2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.alpha_bar = alpha_bar[1:]
            self.alpha_bar_m1 = alpha_bar[:-1]
            self.beta = torch.clamp_max(1 - (self.alpha_bar / self.alpha_bar_m1), 0.999)
            self.alpha = 1 - self.beta
        elif schedule == "linear":
            scale = 1000 / steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            self.beta = torch.linspace(beta_start, beta_end, steps + 1, device=device)
            self.alpha = 1 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
            self.alpha_bar_m1 = torch.cat((torch.ones(1), self.alpha_bar[:-1]), dim=0)
        else:
            raise NotImplementedError("Only linear and cosine scheduling are implemented")
        self.beta_tilde = self.beta * (1 - self.alpha_bar_m1) / (1 - self.alpha_bar)


class GaussianDiffusionProcess(DiffusionProcess):
    def __init__(self, steps: int, schedule: str, s=0.008, device=torch.device('cpu')):
        super(GaussianDiffusionProcess, self).__init__(steps=steps,
                                                       schedule=schedule,
                                                       s=s,
                                                       device=device,
                                                       )

    def x_t(self, x_0, t):
        loc = self.alpha_bar[t].sqrt() * x_0
        eps = torch.randn_like(x_0)
        x_t = loc + eps * (1 - self.alpha_bar[t]).sqrt()
        return x_t, eps

    def mu_bar(self, x_t, x_0, t):
        x_0_weight_num = self.alpha_bar[t - 1].sqrt() * self.beta[t]
        x_0_weight_denom = 1 - self.alpha_bar[t]
        x_0_weight = x_0_weight_num / x_0_weight_denom

        x_t_weight_num = self.alpha_bar[t].sqrt() * (1 - self.alpha_bar[t - 1])
        x_t_weight_denom = 1 - self.alpha_bar[t]
        x_t_weight = x_t_weight_num / x_t_weight_denom

        return x_0_weight * x_0 + x_t_weight * x_t

    def x_tm1(self, x_t, x_0, t):
        loc = self.mu_bar(x_t, x_0, t)
        scale = torch.ones_like(x_0) * self.beta_tilde[t].sqrt()
        return (torch.randn_like(loc) * scale) + loc

    def undiffuse(self, x_t, eps, t):
        # Don't add noise on last step.
        noisescale = self.beta[t].sqrt() if t!=0 else 0
        noise = torch.randn_like(x_t) * noisescale
        eps_scale = (1 - self.alpha[t]) / ((1 - self.alpha_bar[t]).sqrt())
        x_tm1 = self.alpha[t].rsqrt() * (x_t - eps_scale * eps) + noise
        return x_tm1


class CustomStrideGDP(GaussianDiffusionProcess):
    def __init__(self, steps: int, schedule: str, sequence, s=0.008, device=torch.device('cpu')):
        super().__init__(steps, schedule, s=s, device=device)
        self.S = sequence
        # Override alpha/beta vals with strided vals.
        self.alpha = torch.tensor([self.alpha[i] for i in self.S])
        self.alpha_bar = torch.tensor([self.alpha_bar[i] for i in self.S])
        self.beta = 1 - self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.beta_tilde = (1 - self.alpha_bar[:-1]) / (1 - self.alpha_bar[1:])
        self.steps = len(sequence)


class LinearStrideGDP(CustomStrideGDP):
    def __init__(self, steps: int, schedule: str, stride: int, s=0.008, device=torch.device('cpu')):
        sequence = torch.arange(0, steps, stride)
        super().__init__(steps, schedule, sequence, s=s, device=device)


import numpy as np
import torch
import torch.nn.functional as F
from inspect import isfunction
from rotations import *
from denoising_diffusion_pytorch import GaussianDiffusion
from tqdm import tqdm
from distributions import IsotropicGaussianSO3


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class ProjectedGaussianDiffusion(GaussianDiffusion):
    def __init__(self, denoise_fn, timesteps=1000, loss_type='l1', betas = None):
        super().__init__(denoise_fn, image_size=None, timesteps=timesteps, loss_type=loss_type, betas=betas)



    def p_mean_variance(self, x, t, clip_denoised: bool):
        proj_x = self.projection(x)
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(proj_x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, projection):
        self.projection = projection
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, image_size, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        proj_x_noisy = self.projection(x_noisy)
        x_recon = self.denoise_fn(proj_x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, projection, *args, **kwargs):
        self.projection = projection
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)



class ProjectedSO3Diffusion(ProjectedGaussianDiffusion):
    def __init__(self, denoise_fn, timesteps=1000, loss_type='l1', betas = None):
        super().__init__(denoise_fn, timesteps=timesteps, loss_type=loss_type, betas=betas)
        self.register_buffer("identity", torch.eye(3))
    def q_mean_variance(self, x_start, t):
        mean = so3_lerp(self.identity,x_start, extract(self.sqrt_alphas_cumprod, t, x_start.shape))
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_term = so3_scale(x_t, extract(self.sqrt_recip_alphas_cumprod, t, t.shape))
        noise_term = so3_scale(noise, extract(self.sqrt_recipm1_alphas_cumprod, t, t.shape))
        # Translation = subtraction,
        # Rotation = multiply by inverse op (matrices, so transpose)
        return x_t_term @ noise_term.T

    def q_posterior(self, x_start, x_t, t):
        c_1 = so3_scale(x_start, extract(self.posterior_mean_coef1, t, t.shape))
        c_2 = so3_scale(x_t, extract(self.posterior_mean_coef2, t, t.shape))
        posterior_mean = c_1 @ c_2

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        proj_x = self.projection(x)
        predict = self.denoise_fn(proj_x, t)
        # TODO: predict-as-gradient, backprop through projection
        x_recon = self.predict_start_from_noise(x, t=t, noise=predict)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        model_stdev = (0.5 * model_log_variance).exp() * nonzero_mask
        sample = IsotropicGaussianSO3(model_stdev).sample()
        return model_mean @ sample

    @torch.no_grad()
    def p_sample_loop(self, shape, projection):
        self.projection = projection
        device = self.betas.device
        b = shape[0]
        x = IsotropicGaussianSO3(3.0).sample(shape)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long))
        return x

    def q_sample(self, x_start, t, noise=None):
        if noise is not None:
            eps = extract(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
            noise = IsotropicGaussianSO3(eps).sample()

        scale = extract(self.sqrt_alphas_cumprod, t, t.shape)
        x_blend = so3_scale(x_start, scale)
        return x_blend @ noise

    def p_losses(self, x_start, t, noise = None):
        eps = extract(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
        noise = IsotropicGaussianSO3(eps).sample()
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        proj_x_noisy = self.projection(x_noisy)
        x_recon = self.denoise_fn(proj_x_noisy, t)

        descaled_noise = so3_scale(noise, 1/eps)
        distance = rmat_dist(x_recon, descaled_noise)
        loss = (distance ** 2).mean()

        return loss

    def forward(self, x, projection, *args, **kwargs):
        self.projection = projection
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

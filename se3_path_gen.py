import diffusion as diff
from distributions import IGSO3xR3
from util import *

SAMPLES = 100
STEPS = 1000

# q(x_t|x_{t-1}) = N(sqrt(1-B_t)x_(t-1), B_tI)
device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")


def se3_step(x_t: AffineT, beta_t) -> AffineT:
    mean = se3_scale(x_t, torch.sqrt((1 - beta_t)))
    eps = beta_t
    out = IGSO3xR3(eps, mean=mean).sample()
    return out


# Not going to use the denoising function, just getting the betas.
diff_process = diff.SE3Diffusion(None, timesteps=STEPS).to(device)

path = []
# Run 100 samples in parallel
x_t = AffineT(rot=torch.eye(3)[None].expand(SAMPLES,-1,-1), shift=torch.zeros(SAMPLES, 3)).to(device)
for i in range(STEPS):
    path.append(x_t)
    beta_t = diff_process.betas[i]
    x_t = se3_step(x_t, beta_t)
path.append(x_t)
cpu_path = [x.to('cpu') for x in path]
print('aaa')
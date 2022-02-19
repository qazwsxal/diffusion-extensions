import torch
from bingham_train import covpairs, RotPredict, loc
from distributions import Bingham, IsotropicGaussianSO3
from diffusion import SO3Diffusion
from util import *
import pickle
SAMPLES = 2_000
NET_SAMPLES = 1_000
NET_RUNS = SAMPLES//NET_SAMPLES

if __name__ == "__main__":
    import tqdm
    import argparse
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    for _, acro, cov in covpairs:
        results = {}
        for step in tqdm.trange(1000, 100001, 1000):
            net = RotPredict(out_type="skewvec").to(device)
            net.load_state_dict(torch.load(f"weights/weights_bing_{acro}_{step}.pt", map_location=device))
            diff = SO3Diffusion(net, loss_type="skewvec").to(device)

            bing = Bingham(loc=loc.to(device), covariance_matrix=cov.to(device))
            bing_samples =  bing.sample((SAMPLES,))
            bing_samples =quat_to_rmat(bing_samples)
            diff_start = IsotropicGaussianSO3(eps=torch.ones(1, device=device))
            diff_samples = []
            for i in tqdm.trange(NET_RUNS):
                R = diff.p_sample_loop((NET_SAMPLES,))
                diff_samples.append(R)
            diff_samples = torch.cat(diff_samples, dim=0)
            print('aaaa')
            mmd = MMD(bing_samples, diff_samples, rmat_gaussian_kernel)
            results[step] = mmd.item()
        pickle.dump(results, open(f'bingham_mmd_{acro}.pkl', 'wb'))
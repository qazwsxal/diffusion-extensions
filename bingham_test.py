import torch
from bingham_train import covpairs, RotPredict, loc
from distributions import Bingham, IsotropicGaussianSO3
from diffusion import SO3Diffusion
from util import *
import pickle
SAMPLES = 100_000
NET_SAMPLES = 100_000
NET_RUNS = SAMPLES//NET_SAMPLES
device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")


def calc_step(acro, cov, step):
    net = RotPredict(out_type="skewvec").to(device)
    net.load_state_dict(torch.load(f"weights/weights_bing_{acro}_{step}.pt", map_location=device))
    diff = SO3Diffusion(net, loss_type="skewvec").to(device)

    bing = Bingham(loc=loc.to(device), covariance_matrix=cov.to(device))
    bing_samples = bing.sample((SAMPLES,))
    bing_samples = quat_to_rmat(bing_samples)
    diff_samples = []
    for i in range(NET_RUNS):
        R = diff.p_sample_loop((NET_SAMPLES,))
        diff_samples.append(R)
    diff_samples = torch.cat(diff_samples, dim=0)
    print('aaaa')
    mmd = MMD(bing_samples, diff_samples, rmat_gaussian_kernel, chunksize=4_000)
    return mmd.item()

if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp
    mp.set_start_method('spawn') # need so cuda can be used in multiple threads
    parser = argparse.ArgumentParser(description="Aircraft rotation args")
    parser.add_argument(
        "cov", type=str, help="covariance matrix to use", choices = ["sur", "scr", "lur", "lcr"]
    )
    args = parser.parse_args()
    acro = args.cov
    cov, = [c for _, a, c in covpairs if a == acro]
    results = dict()
    eval_points = [(acro, cov, step) for step in [100_000,]]
    with mp.Pool(processes=2) as pool:
        p_results = pool.starmap(calc_step, eval_points)
    for (acro, cov, step), mmd in zip(eval_points, p_results):
        results[step] = mmd
    pickle.dump(results, open(f'bingham_mmd_{acro}.pkl', 'wb'))
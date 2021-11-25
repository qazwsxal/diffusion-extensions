from torch.utils.data import DataLoader

from models import ProtNet
from prot_util import *
from util import identity, to_device, init_from_dict
from diffusion import ProjectedSE3Diffusion
from itertools import count


AUGMENT = True

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    import wandb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch", type=int, default=1, help="batch size"
        )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate",
        )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="transformer dimension",
        )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="number of self-attention heads per layer",
        )
    parser.add_argument(
        "--dim_head",
        type=int,
        default=64,
        help="dimension of self-attention head",
        )
    parser.add_argument(
        "--num_neighbours",
        type=int,
        default=8,
        help="number of neighbours for SE3 Transformer",
        )
    parser.add_argument(
        "--t_depth",
        type=int,
        default=3,
        help="number of transformer layers",
        )
    parser.add_argument(
        "--c_depth",
        type=int,
        default=3,
        help="number of residue convolutional layers",
        )

    args = parser.parse_args()
    wandb.init(project="ProtDiffusion", entity="qazwsxal", config=args)

    config = wandb.config


    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = ProtDataset("data/BPTI_dock")
    dl = DataLoader(dataset, batch_size=config["batch"], shuffle=True, num_workers=4, pin_memory=True,
                    collate_fn=identity)

    net, = init_from_dict(config, ProtNet)
    net.to(device)
    net.train()
    wandb.watch(net, log_freq=100)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    diff_model = ProjectedSE3Diffusion(net).to(device)

    true_rot = torch.eye(3).unsqueeze(0).expand(config["batch"], -1, -1).to(device)
    true_shift = torch.zeros(config["batch"], 3).to(device)
    true_pos = AffineT(shift=true_shift, rot=true_rot)
    for epoch in count():
        for data in dl:
            ...
            data = to_device(device, *data)
            # Random transform.
            if AUGMENT:
                transl = torch.randn(((config["batch"]), 3)).to(device)
                rot = torch.linalg.qr(torch.randn(((config["batch"]), 3, 3)))[0].to(device)
                aff_ts = [AffineT(shift=t, rot=r) for t,r in zip(transl, rot)]
                data = [move_prots(t, p) for t,p in zip(aff_ts, data)]
            projection = ProtProjection(data).to(device)


            loss = diff_model(true_pos, projection)
            loss.backward()
            wandb.log({"loss": loss.item()})
        optim.step()
        optim.zero_grad()
        torch.save(net.state_dict(), "weights/weights_protein.pt")

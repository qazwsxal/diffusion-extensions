import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch.utils.data import DataLoader

from datasets import ShapeNet
from util import skew2vec, log_rmat


def draw_rot_grads(rot, grads, fig=None):
    rot = rot.detach().cpu()
    grads = grads.detach().cpu()
    zeros = torch.zeros(3)
    fig = plt.figure(num=fig)
    ax = fig.add_subplot(111, projection='3d')
    maxval = max((rot + grads).max(), -(rot + grads).min())
    ax.set_xlim(-maxval, maxval)
    ax.set_ylim(-maxval, maxval)
    ax.set_zlim(-maxval, maxval)

    for i in range(3):
        col = rot[:, i]
        axis = torch.stack((zeros, col), dim=-1)
        gradstart = col
        gradend = col + grads[:, i]
        grad = torch.stack((gradstart,gradend), dim=-1)
        ax.plot(*axis.numpy())
        ax.plot(*grad.numpy())
    fig.show()



if __name__ == "__main__":
    import wandb

    wandb.init(project='GradTest', entity='qazwsxal')
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = ShapeNet('train', (0,))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    data = next(iter(dl))
    data.requires_grad = True
    pred_grads = torch.randn_like(data)
    pred_grads.requires_grad = True
    optim = torch.optim.Adam((pred_grads,), lr=1e-5)

    rot = torch.tensor([[[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]]], dtype=torch.float32)
    log_rot = log_rmat(rot)
    rot_grad = (log_rot @ rot)
    skew_targ = skew2vec(log_rot)

    rot.requires_grad = True

    draw_rot_grads(rot[0], rot_grad[0])

    i = 0
    while i < 10000:
        proj_data = (rot @ data.transpose(-1, -2)).transpose(-1, -2)
        proj_grads = (rot_grad @ data.transpose(-1, -2)).transpose(-1, -2)

        orth_loss = (proj_data * pred_grads).sum(dim=-1).pow(2).mean()
        r_grad = torch.autograd.grad(proj_data, rot, proj_grads, create_graph=True)[0]
        draw_rot_grads(rot[0], r_grad[0])
        s_v = r_grad @ rot.transpose(-1, -2)
        # Extract skew-symmetric part i.e. project onto tangent
        s_v_proj = (s_v - s_v.transpose(-1, -2)) / 2
        sym_part = (s_v + s_v.transpose(-1, -2)) / 2
        sym_loss = sym_part.pow(2).mean()
        # Convert to vector form for regression
        predict = skew2vec(s_v_proj)

        loss = F.mse_loss(predict, skew_targ) + sym_loss + orth_loss
        print(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        logdict = {"loss": loss.detach()}
        # Initial setup and prediction of some test data.
        if i % 100 == 0:
            start = proj_data[0].detach().cpu().numpy()[::4] / 2
            end = (proj_data[0] - 0.1 * proj_grads[0]).detach().cpu().numpy()[::4] / 2

            logdict["Predicted Gradients"] = wandb.Object3D(
                {"type": "lidar/beta",
                 "points": proj_data[0].detach().cpu().numpy() / 2,
                 "vectors": np.array([{"start": s.tolist(), "end": e.tolist()}
                                      for s, e in zip(start, end)
                                      ]),
                 "boxes": np.array([{
                     "corners": [
                         [-0.5, -0.5, -0.5],
                         [-0.5, 0.5, -0.5],
                         [-0.5, -0.5, 0.5],
                         [0.5, -0.5, -0.5],
                         [0.5, 0.5, -0.5],
                         [-0.5, 0.5, 0.5],
                         [0.5, -0.5, 0.5],
                         [0.5, 0.5, 0.5]
                         ],
                     # "label": "Tree",
                     "color": [123, 321, 111],
                     }, ])
                 }
                )
            logdict["True Position"] = wandb.Object3D(
                {"type": "lidar/beta",
                 "points": data[0].detach().cpu().numpy() / 2
                 }
                )
        i += 1
        wandb.log(logdict)
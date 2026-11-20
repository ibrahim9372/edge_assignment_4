"""
CIFAR-10 / CIFAR-100 data pipeline.

The whole dataset is small (50k x 3 x 32 x 32 uint8 = ~150 MB), so we keep it
resident on the GPU as uint8 and do augmentation (random-crop + horizontal
flip) and normalisation directly on the GPU. This removes the DataLoader /
worker-process bottleneck (which is especially slow on Windows) and keeps the
GPU busy.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# Per-channel statistics of the training sets (standard values).
MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def load_cifar(name: str, root: str = "data"):
    """Download (if needed) and return ((x_train, y_train), (x_test, y_test), class_names).

    Images are returned as uint8 tensors of shape (N, 3, 32, 32).
    Uses data/<name>.npz if it exists (see tools/prepare_data.py), else torchvision."""
    npz = os.path.join(root, f"{name}.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        to = lambda x, y: (torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y).long())
        return to(d["x_train"], d["y_train"]), to(d["x_test"], d["y_test"]), [str(c) for c in d["classes"]]
    ds_cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    train = ds_cls(root, train=True, download=True)
    test = ds_cls(root, train=False, download=True)

    def to_tensor(ds):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        y = torch.tensor(ds.targets, dtype=torch.long)
        return x, y

    return to_tensor(train), to_tensor(test), list(train.classes)


def train_val_split(x, y, val_size: int = 5000, seed: int = 0):
    """Deterministic hold-out split of the training set (used by NAS so that the
    test set is never touched during the search)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    return (x[tr_idx], y[tr_idx]), (x[val_idx], y[val_idx])


def _random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Vectorised random-crop (with zero padding) + random horizontal flip on a GPU batch."""
    b, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None, :])  # (B, H)
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])  # (B, W)
    bidx = torch.arange(b, device=x.device)[:, None, None]
    # Advanced indexing -> (B, H, W, C); permute back to NCHW.
    out = xp.permute(0, 2, 3, 1)[bidx, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=x.device) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out


class GPULoader:
    """Minimal iterable that yields normalised (x, y) batches from GPU-resident data."""

    def __init__(self, x_u8, y, batch_size, dataset, train, device="cuda",
                 resolution: int = 32, drop_last: bool = False):
        self.x = x_u8.to(device)
        self.y = y.to(device)
        self.bs = batch_size
        self.train = train
        self.device = device
        self.res = resolution
        self.drop_last = drop_last
        self.mean = torch.tensor(MEAN[dataset], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD[dataset], device=device).view(1, 3, 1, 1)

    def __len__(self):
        n = len(self.x)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, device=self.device) if self.train else torch.arange(n, device=self.device)
        for i in range(0, n, self.bs):
            b = idx[i:i + self.bs]
            if self.drop_last and len(b) < self.bs:
                break
            x = self.x[b].float().div_(255.0)
            if self.train:
                x = _random_crop_flip(x)
            if self.res != 32:  # variable input resolution (used by NAS)
                x = F.interpolate(x, size=(self.res, self.res), mode="bilinear",
                                  align_corners=False, antialias=True)
            x = (x - self.mean) / self.std
            # .contiguous(): the crop above yields an NHWC-strided tensor; cuDNN on this Turing GPU
            # was ~15x slower with channels_last layout, so force standard NCHW memory layout.
            yield x.contiguous(), self.y[b]

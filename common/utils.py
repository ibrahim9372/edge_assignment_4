"""Small shared helpers: seeding, checkpoints, CSV logging, parameter / MAC counting."""
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed input sizes -> let cuDNN pick fastest kernels


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, input_shape=(1, 3, 32, 32)) -> int:
    """Multiply-accumulate count of Conv2d + Linear layers for one image (forward hooks)."""
    macs = [0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        macs[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * k

    def lin_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(input_shape, device=dev))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs[0]


def save_checkpoint(path, model, arch, num_classes, dataset, classes, resolution=32,
                    config=None, metrics=None, args=None, half=False):
    """Save a self-describing checkpoint so any script can rebuild the model from it.

    half=True stores floating-point weights in fp16 (used for the large ResNet-50
    teacher to keep the file < 50 MB for GitHub); they are cast back on load."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = model.state_dict()
    if half:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save({
        "arch": arch, "num_classes": num_classes, "dataset": dataset, "classes": classes,
        "resolution": resolution, "config": config, "metrics": metrics or {},
        "args": args or {}, "state_dict": {k: v.cpu() for k, v in sd.items()},
    }, path)


class CSVLogger:
    """Append-per-epoch CSV logger (header written on first row)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.header = None
        if os.path.exists(path):
            os.remove(path)

    def log(self, row: dict):
        new = self.header is None
        if new:
            self.header = list(row.keys())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header)
            if new:
                w.writeheader()
            w.writerow(row)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

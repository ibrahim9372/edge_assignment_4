"""Shared pre-processing for the PyTorch and ONNX Runtime inference pipelines (pure NumPy,
so the ONNX path has no dependency on PyTorch)."""
import glob
import os

import numpy as np
from PIL import Image

MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}

# Every deployable model of the project (students: baseline + KD, and the NAS model).
MODEL_GLOBS = ["checkpoints/baseline/*.pt", "checkpoints/kd/*.pt", "checkpoints/nas/*.pt"]


def all_checkpoints(include_teacher=False):
    paths = sorted(p for g in MODEL_GLOBS for p in glob.glob(g))
    if include_teacher:
        paths = sorted(glob.glob("checkpoints/teacher/*.pt")) + paths
    return [p.replace("\\", "/") for p in paths]


def model_name(path):
    """checkpoints/kd/mobilenetv2_cifar100.pt -> kd/mobilenetv2_cifar100"""
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}/{os.path.splitext(os.path.basename(path))[0]}"


def preprocess_uint8(batch_u8_nchw: np.ndarray, dataset: str, resolution: int) -> np.ndarray:
    """uint8 (N,3,32,32) -> normalised float32 (N,3,R,R)."""
    x = batch_u8_nchw.astype(np.float32) / 255.0
    if resolution != x.shape[-1]:
        x = np.stack([_resize_chw(im, resolution) for im in x])
    m = np.array(MEAN[dataset], np.float32).reshape(1, 3, 1, 1)
    s = np.array(STD[dataset], np.float32).reshape(1, 3, 1, 1)
    return ((x - m) / s).astype(np.float32)


def _resize_chw(im, r):
    pil = Image.fromarray((im.transpose(1, 2, 0) * 255).astype(np.uint8)).resize((r, r), Image.BILINEAR)
    return np.asarray(pil, np.float32).transpose(2, 0, 1) / 255.0


def load_image(path: str, dataset: str, resolution: int) -> np.ndarray:
    """Any image file -> (1,3,R,R) float32 tensor (resized like CIFAR: 32x32 first)."""
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img, np.uint8).transpose(2, 0, 1)[None]
    return preprocess_uint8(arr, dataset, resolution)


def load_testset(dataset: str, root="data"):
    """CIFAR test set as uint8 NCHW numpy + labels (prepared .npz if present, else torchvision)."""
    npz = os.path.join(root, f"{dataset}.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["x_test"].transpose(0, 3, 1, 2), d["y_test"]
    import torchvision
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    ds = cls(root, train=False, download=True)
    return ds.data.transpose(0, 3, 1, 2), np.array(ds.targets)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

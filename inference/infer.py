"""
Q1.6 / deliverable 9 — Unified PyTorch inference pipeline for every trained model
(MobileNetV2 / SqueezeNet / EfficientNet-Lite0 — baseline and distilled — and the NAS model).

All checkpoints are self-describing (arch, dataset, classes, input resolution, NAS config),
so one script rebuilds and runs any of them.

Examples
    # classify an image with one model
    python -m inference.infer --ckpt checkpoints/kd/mobilenetv2_cifar100.pt --image samples/cat.png
    # evaluate every model on its test set (CIFAR-100 students, CIFAR-10 NAS model)
    python -m inference.infer --all --testset
"""
import argparse
import time

import numpy as np
import torch

from common.registry import load_checkpoint
from inference.common_io import all_checkpoints, load_image, load_testset, model_name, preprocess_uint8, softmax


class Predictor:
    """Wraps any project checkpoint behind one predict() API."""

    def __init__(self, ckpt_path, device="cpu"):
        self.model, meta = load_checkpoint(ckpt_path, device)
        self.device = device
        self.name = model_name(ckpt_path)
        self.dataset, self.classes, self.res = meta["dataset"], meta["classes"], meta["resolution"]
        self.arch = meta["arch"]

    @torch.inference_mode()
    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.model(torch.from_numpy(x).to(self.device)).float().cpu().numpy()

    def predict_image(self, path, topk=5):
        p = softmax(self.logits(load_image(path, self.dataset, self.res)))[0]
        idx = p.argsort()[::-1][:topk]
        return [(self.classes[i], float(p[i])) for i in idx]

    def evaluate(self, bs=500):
        x_u8, y = load_testset(self.dataset)
        correct, t = 0, 0.0
        for i in range(0, len(y), bs):
            x = preprocess_uint8(x_u8[i:i + bs], self.dataset, self.res)
            t0 = time.perf_counter()
            pred = self.logits(x).argmax(1)
            t += time.perf_counter() - t0
            correct += (pred == y[i:i + bs]).sum()
        return 100.0 * correct / len(y), len(y) / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[], help="checkpoint path(s)")
    ap.add_argument("--all", action="store_true", help="use every student + NAS checkpoint")
    ap.add_argument("--image", nargs="*", default=[], help="image file(s) to classify")
    ap.add_argument("--testset", action="store_true", help="report test accuracy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = all_checkpoints() if args.all else args.ckpt
    if not paths:
        ap.error("give --ckpt or --all")
    print(f"{'model':40s} {'arch':20s} {'dataset':9s} {'top1 %':>7s} {'img/s':>9s}")
    for p in paths:
        pr = Predictor(p, args.device)
        for img in args.image:
            preds = ", ".join(f"{c} {s:.2f}" for c, s in pr.predict_image(img))
            print(f"[{pr.name}] {img}: {preds}")
        if args.testset:
            acc, ips = pr.evaluate()
            print(f"{pr.name:40s} {pr.arch:20s} {pr.dataset:9s} {acc:7.2f} {ips:9.0f}")


if __name__ == "__main__":
    main()

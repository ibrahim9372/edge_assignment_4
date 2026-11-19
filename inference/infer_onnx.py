"""
Q1.7 / Q2.8 / deliverable 10 — CPU inference with ONNX Runtime (no PyTorch needed at runtime
except for downloading the test set).

    python -m inference.infer_onnx --onnx onnx/kd/mobilenetv2_cifar100.onnx --image samples/cat.png
    python -m inference.infer_onnx --all --testset --threads 4
"""
import argparse
import glob
import json
import time

import numpy as np
import onnxruntime as ort

from inference.common_io import load_image, load_testset, preprocess_uint8, softmax


class ORTPredictor:
    def __init__(self, onnx_path, threads=1):
        with open(onnx_path.replace(".onnx", ".json")) as f:
            self.meta = json.load(f)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.name = self.meta["name"]

    def logits(self, x):
        return self.sess.run(None, {self.inp: x})[0]

    def predict_image(self, path, topk=5):
        x = load_image(path, self.meta["dataset"], self.meta["resolution"])
        p = softmax(self.logits(x))[0]
        idx = p.argsort()[::-1][:topk]
        return [(self.meta["classes"][i], float(p[i])) for i in idx]

    def evaluate(self, bs=250):
        x_u8, y = load_testset(self.meta["dataset"])
        correct, t = 0, 0.0
        for i in range(0, len(y), bs):
            x = preprocess_uint8(x_u8[i:i + bs], self.meta["dataset"], self.meta["resolution"])
            t0 = time.perf_counter()
            pred = self.logits(x).argmax(1)
            t += time.perf_counter() - t0
            correct += (pred == y[i:i + bs]).sum()
        return 100.0 * correct / len(y), len(y) / t


def all_onnx():
    return sorted(p.replace("\\", "/") for g in ("baseline", "kd", "nas") for p in glob.glob(f"onnx/{g}/*.onnx"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--image", nargs="*", default=[])
    ap.add_argument("--testset", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    paths = all_onnx() if args.all else args.onnx
    if not paths:
        ap.error("give --onnx or --all")
    print(f"ONNX Runtime {ort.__version__} | CPUExecutionProvider | {args.threads} thread(s)")
    for p in paths:
        pr = ORTPredictor(p, args.threads)
        for img in args.image:
            print(f"[{pr.name}] {img}: " + ", ".join(f"{c} {s:.2f}" for c, s in pr.predict_image(img)))
        if args.testset:
            acc, ips = pr.evaluate()
            print(f"{pr.name:40s} {pr.meta['dataset']:9s} top1={acc:6.2f}%  {ips:8.0f} img/s")


if __name__ == "__main__":
    main()

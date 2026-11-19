"""
Q1.7 / Q2.8 / deliverable 10 — Export trained models to ONNX.

For each checkpoint:  onnx/<group>/<name>.onnx  (opset 17, dynamic batch dimension)
                     onnx/<group>/<name>.json  (metadata sidecar: dataset, resolution, classes)
After export the ONNX graph is checked with onnx.checker and its outputs are compared
with PyTorch on a random batch (max |difference| reported).

    python -m inference.export_onnx --all
    python -m inference.export_onnx --ckpt checkpoints/nas/nas_best_cifar10.pt
"""
import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common.registry import load_checkpoint
from inference.common_io import all_checkpoints, model_name


def export(ckpt_path, out_root="onnx", opset=17):
    model, meta = load_checkpoint(ckpt_path, "cpu")
    name = model_name(ckpt_path)
    out = os.path.join(out_root, name + ".onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = meta["resolution"]
    dummy = torch.randn(2, 3, r, r)
    torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(out))

    # numerical check vs PyTorch
    x = torch.randn(8, 3, r, r)
    with torch.inference_mode():
        ref = model(x).numpy()
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())

    with open(out.replace(".onnx", ".json"), "w") as f:
        json.dump({"name": name, "arch": meta["arch"], "dataset": meta["dataset"], "resolution": r,
                   "classes": meta["classes"], "config": meta.get("config"),
                   "pytorch_metrics": meta.get("metrics")}, f, indent=2)
    print(f"{name:40s} -> {out}  ({os.path.getsize(out) / 2**20:.2f} MB, max|torch-ort|={diff:.2e})")
    return out, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--include-teacher", action="store_true")
    args = ap.parse_args()
    paths = all_checkpoints(args.include_teacher) if args.all else args.ckpt
    for p in paths:
        export(p)


if __name__ == "__main__":
    main()

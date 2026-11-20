"""
Deliverable 11 — Automated benchmarking of every trained model.

For each checkpoint (teacher, 3 baseline students, 3 KD students, NAS model) it records:
  accuracy  : PyTorch test top-1 / top-5, ONNX Runtime test top-1 (checks export fidelity)
  size      : parameters, MMACs, checkpoint size, ONNX size
  CPU speed : PyTorch latency (batch 1, 1 thread), ONNX Runtime latency (batch 1, 1 thread),
              ONNX Runtime throughput (batch 32, all physical cores)
  memory    : peak inference RSS increase (PyTorch and ONNX Runtime, batch 1)
All CPU measurements run in isolated subprocesses (common/profiler.py).

Outputs: results/benchmarks.csv, results/q1/kd_vs_baseline.csv

    python -m benchmark.run_benchmarks
"""
import argparse
import os

import pandas as pd
import psutil
import torch

from common.data import GPULoader, load_cifar
from common.profiler import profile_in_subprocess
from common.registry import load_checkpoint
from common.train_loop import evaluate
from common.utils import count_macs, count_params
from inference.common_io import all_checkpoints, model_name
from inference.infer_onnx import ORTPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default="results/benchmarks.csv")
    args = ap.parse_args()
    cores = psutil.cpu_count(logical=False)

    test = {}
    rows = []
    for ck in all_checkpoints(include_teacher=True):
        model, meta = load_checkpoint(ck, "cuda")
        ds, r = meta["dataset"], meta["resolution"]
        if ds not in test:
            _, (xte, yte), _ = load_cifar(ds)
            test[ds] = (xte, yte)
        name = model_name(ck)
        group = name.split("/")[0]
        print(f"--- {name}", flush=True)
        acc = evaluate(model,
                       GPULoader(*test[ds], 500, ds, train=False, resolution=r))
        row = {"model": name, "group": group, "arch": meta["arch"], "dataset": ds, "resolution": r,
               "top1": round(acc["top1"], 2), "top5": round(acc["top5"], 2),
               "params_M": round(count_params(model) / 1e6, 3),
               "MMACs": round(count_macs(model, (1, 3, r, r)) / 1e6, 1),
               "ckpt_MB": round(os.path.getsize(ck) / 2**20, 2)}
        del model
        torch.cuda.empty_cache()

        pt = profile_in_subprocess({"backend": "torch", "ckpt": ck, "resolution": r, "threads": 1,
                                    "iters": args.iters})
        row.update({"fp32_weights_MB": round(pt["size_mb"], 2),
                    "pt_cpu_latency_ms": round(pt["latency_ms_median"], 3),
                    "pt_cpu_latency_p95_ms": round(pt["latency_ms_p95"], 3),
                    "pt_peak_mem_MB": round(pt["peak_mem_mb"], 1)})

        onnx_path = f"onnx/{name}.onnx"
        if os.path.exists(onnx_path):
            o1 = profile_in_subprocess({"backend": "ort", "onnx": onnx_path, "resolution": r, "threads": 1,
                                        "iters": args.iters})
            ob = profile_in_subprocess({"backend": "ort", "onnx": onnx_path, "resolution": r, "threads": cores,
                                        "batch": 32, "iters": 30})
            ort_acc, _ = ORTPredictor(onnx_path, threads=cores).evaluate()
            row.update({"onnx_MB": round(o1["size_mb"], 2), "ort_top1": round(float(ort_acc), 2),
                        "ort_cpu_latency_ms": round(o1["latency_ms_median"], 3),
                        "ort_cpu_latency_p95_ms": round(o1["latency_ms_p95"], 3),
                        "ort_peak_mem_MB": round(o1["peak_mem_mb"], 1),
                        f"ort_throughput_b32_{cores}t_img_s": round(ob["throughput_img_s"], 0),
                        "ort_speedup_vs_pt": round(pt["latency_ms_median"] / o1["latency_ms_median"], 2)})
        print({k: v for k, v in row.items() if k not in ("model",)}, flush=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n", df.to_string(index=False))

    # Baseline vs KD comparison table for Q1
    b = df[df.group == "baseline"].set_index("arch")
    k = df[df.group == "kd"].set_index("arch")
    common_archs = [a for a in b.index if a in k.index]
    if common_archs:
        teacher_top1 = df[df.group == "teacher"].top1.max() if (df.group == "teacher").any() else None
        cmp = pd.DataFrame({"arch": common_archs,
                            "teacher_top1": teacher_top1,
                            "baseline_top1": [b.loc[a, "top1"] for a in common_archs],
                            "kd_top1": [k.loc[a, "top1"] for a in common_archs],
                            "baseline_top5": [b.loc[a, "top5"] for a in common_archs],
                            "kd_top5": [k.loc[a, "top5"] for a in common_archs],
                            "params_M": [k.loc[a, "params_M"] for a in common_archs],
                            "ort_cpu_latency_ms": [k.loc[a].get("ort_cpu_latency_ms") for a in common_archs]})
        cmp["kd_gain_top1"] = (cmp.kd_top1 - cmp.baseline_top1).round(2)
        cmp.to_csv("results/q1/kd_vs_baseline.csv", index=False)
        print("\n", cmp.to_string(index=False))


if __name__ == "__main__":
    main()

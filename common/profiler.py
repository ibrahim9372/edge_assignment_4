"""
Hardware profiler (CPU) used by the hardware-aware NAS (Q2.5/Q2.6) and the benchmarks.

Measures, for a model on the CPU:
  * inference latency  — real wall-clock timing (warm-up, then median / mean / p95 over N runs)
  * storage size       — bytes of the serialised fp32 state_dict (PyTorch) or of the .onnx file
  * peak inference memory — increase of the process resident set size (RSS) during inference,
                         sampled by a background thread with psutil (supported on Windows,
                         Linux and macOS; reported as None if psutil is unavailable).

Each measurement runs in a *fresh subprocess* so results are isolated from the training
process (no CUDA context, no cached allocator memory, fixed thread count).

    python -m common.profiler --spec '{"backend": "torch", "arch": "mobilenetv2", "num_classes": 100}'
"""
import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = "PROFILE_RESULT "


class PeakRSSSampler:
    """Background thread that records the maximum RSS of this process."""

    def __init__(self, interval=0.0005):
        try:
            import psutil
            self.proc = psutil.Process()
        except ImportError:
            self.proc = None
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def rss(self):
        return self.proc.memory_info().rss if self.proc else 0

    def __enter__(self):
        self.base = self.rss()
        self.peak = self.base
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.rss())
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()
        self.peak = max(self.peak, self.rss())

    @property
    def delta_mb(self):
        return None if self.proc is None else (self.peak - self.base) / 2**20


def _time_runs(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"latency_ms_median": float(np.median(ts)), "latency_ms_mean": float(ts.mean()),
            "latency_ms_p95": float(np.percentile(ts, 95))}


def run_profile(spec: dict) -> dict:
    """Executed inside the worker subprocess."""
    threads = spec.get("threads", 1)
    batch = spec.get("batch", 1)
    res = spec.get("resolution", 32)
    warmup, iters = spec.get("warmup", 10), spec.get("iters", 50)
    x = np.random.rand(batch, 3, res, res).astype(np.float32)
    out = {"backend": spec["backend"], "threads": threads, "batch": batch, "resolution": res}

    if spec["backend"] == "ort":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(spec["onnx"], so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        out["size_mb"] = os.path.getsize(spec["onnx"]) / 2**20
        fn = lambda: sess.run(None, {name: x})
    else:
        import torch
        torch.set_num_threads(threads)
        sys.path.insert(0, ROOT)
        from common.registry import build_model, load_checkpoint
        if spec.get("ckpt"):
            model, _ = load_checkpoint(spec["ckpt"], "cpu")
        else:  # untrained weights are fine for latency / size / memory
            model = build_model(spec["arch"], spec["num_classes"], spec.get("config")).eval()
        buf = io.BytesIO()
        torch.save({k: v.float() if v.is_floating_point() else v for k, v in model.state_dict().items()}, buf)
        out["size_mb"] = buf.tell() / 2**20
        out["params"] = sum(p.numel() for p in model.parameters())
        xt = torch.from_numpy(x)

        def fn():
            with torch.inference_mode():
                model(xt)

    with PeakRSSSampler() as s:
        out.update(_time_runs(fn, warmup, iters))
    out["peak_mem_mb"] = s.delta_mb
    out["throughput_img_s"] = 1000.0 * batch / out["latency_ms_median"]
    return out


def profile_in_subprocess(spec: dict, timeout=600) -> dict:
    """Launch a clean Python process that runs `run_profile(spec)` and return its result."""
    proc = subprocess.run([sys.executable, "-m", "common.profiler", "--spec", json.dumps(spec)],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    for line in proc.stdout.splitlines():
        if line.startswith(TAG):
            return json.loads(line[len(TAG):])
    raise RuntimeError(f"profiling failed:\n{proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON profiling specification")
    result = run_profile(json.loads(ap.parse_args().spec))
    print(TAG + json.dumps(result), flush=True)

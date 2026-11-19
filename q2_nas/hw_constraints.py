"""
Q2.5 / Q2.6 — Hardware-aware, resource-constrained candidate selection.

Every candidate is profiled on the actual CPU of this machine *before* any training:
  1. latency on CPU (batch 1, single thread — emulating a small edge core),
  2. model storage size (serialised fp32 weights),
  3. peak inference memory (RSS increase during inference; where psutil is supported).
Candidates violating any deployment constraint are rejected (and cost no training time).
"""
from dataclasses import asdict, dataclass

from common.profiler import profile_in_subprocess


@dataclass
class DeploymentConstraints:
    max_latency_ms: float = 2.0     # median batch-1 latency, 1 CPU thread
    max_size_mb: float = 1.0        # fp32 weight storage
    max_peak_mem_mb: float = 64.0   # extra RSS during inference

    def to_dict(self):
        return asdict(self)


def profile_candidate(cfg: dict, num_classes: int = 10, threads: int = 1) -> dict:
    return profile_in_subprocess({"backend": "torch", "arch": "nas", "num_classes": num_classes,
                                  "config": cfg, "resolution": cfg["resolution"], "threads": threads,
                                  "warmup": 10, "iters": 40})


def check_constraints(hw: dict, c: DeploymentConstraints):
    """Returns (feasible, list of violation reasons)."""
    reasons = []
    if hw["latency_ms_median"] > c.max_latency_ms:
        reasons.append(f"latency {hw['latency_ms_median']:.2f}ms > {c.max_latency_ms}ms")
    if hw["size_mb"] > c.max_size_mb:
        reasons.append(f"size {hw['size_mb']:.2f}MB > {c.max_size_mb}MB")
    if hw.get("peak_mem_mb") is not None and hw["peak_mem_mb"] > c.max_peak_mem_mb:
        reasons.append(f"peak mem {hw['peak_mem_mb']:.1f}MB > {c.max_peak_mem_mb}MB")
    return len(reasons) == 0, reasons

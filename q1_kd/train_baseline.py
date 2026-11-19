"""
Q1.5 — Baseline pipeline: train MobileNetV2, SqueezeNet and EfficientNet-Lite0 on
CIFAR-100 with hard-label cross-entropy only (no teacher is loaded at all).

    python -m q1_kd.train_baseline [--arch all] [--epochs 40]
"""
import sys

from q1_kd.train_students import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "baseline"]
    main()

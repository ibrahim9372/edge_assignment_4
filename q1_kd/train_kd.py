"""
Q1.4 — Distillation pipeline: train MobileNetV2, SqueezeNet and EfficientNet-Lite0
on CIFAR-100 from the frozen ResNet-50 teacher (hard CE + temperature soft targets +
feature/hint loss).

    python -m q1_kd.train_kd [--arch all] [--T 4] [--alpha 0.3] [--beta 1]
"""
import sys

from q1_kd.train_students import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "kd"]
    main()

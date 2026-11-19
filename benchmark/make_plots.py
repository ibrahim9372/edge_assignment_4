"""
Generate report figures from the CSV/JSON results.

    python -m benchmark.make_plots
Writes PNGs to results/figures/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results/figures"
BLUE, ORANGE, GREEN, GREY, RED = "#2a6fdb", "#e8862a", "#2e9e5b", "#9aa0a6", "#d64545"


def savefig(name):
    os.makedirs(OUT, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}", dpi=150)
    plt.close()
    print("saved", f"{OUT}/{name}")


def kd_bar():
    p = "results/q1/kd_vs_baseline.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    x = range(len(d))
    plt.figure(figsize=(7, 4))
    b1 = plt.bar([i - 0.2 for i in x], d.baseline_top1, 0.4, label="Baseline (CE only)", color=GREY)
    b2 = plt.bar([i + 0.2 for i in x], d.kd_top1, 0.4, label="Distilled (KD)", color=BLUE)
    for bars in (b1, b2):
        for b in bars:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}",
                     ha="center", fontsize=9)
    if d.teacher_top1.notna().any():
        plt.axhline(d.teacher_top1.iloc[0], ls="--", color=ORANGE, label=f"ResNet-50 teacher ({d.teacher_top1.iloc[0]:.1f})")
    plt.xticks(list(x), d.arch)
    plt.ylabel("CIFAR-100 test top-1 (%)")
    plt.ylim(max(0, d[["baseline_top1", "kd_top1"]].min().min() - 15), 100)
    plt.title("Knowledge distillation: baseline vs distilled students")
    plt.legend(loc="upper right", fontsize=8)
    savefig("q1_kd_vs_baseline.png")


def training_curves():
    plt.figure(figsize=(8, 4.5))
    colors = {"mobilenetv2": BLUE, "squeezenet": GREEN, "efficientnet_lite0": ORANGE}
    for arch, c in colors.items():
        for mode, ls in (("baseline", ":"), ("kd", "-")):
            p = f"results/q1/logs/{mode}_{arch}.csv"
            if os.path.exists(p):
                d = pd.read_csv(p)
                plt.plot(d.epoch, d.eval_top1, ls, color=c, label=f"{arch} ({mode})")
    p = "results/q1/logs/teacher_resnet50.csv"
    if os.path.exists(p):
        d = pd.read_csv(p)
        plt.plot(d.epoch, d.eval_top1, "-", color="black", label="ResNet-50 teacher")
    plt.xlabel("epoch")
    plt.ylabel("test top-1 (%)")
    plt.title("CIFAR-100 learning curves")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.3)
    savefig("q1_learning_curves.png")


def latency_vs_acc():
    p = "results/benchmarks.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.dataset == "cifar100"]
    plt.figure(figsize=(7, 4.5))
    for _, r in d.iterrows():
        c = {"teacher": ORANGE, "baseline": GREY, "kd": BLUE}.get(r.group, GREEN)
        plt.scatter(r.pt_cpu_latency_ms, r.top1, s=40 + 8 * r.params_M, color=c)
        plt.annotate(f"{r.arch} ({r.group})", (r.pt_cpu_latency_ms, r.top1), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("PyTorch CPU latency, batch 1, 1 thread (ms, log)")
    plt.ylabel("CIFAR-100 top-1 (%)")
    plt.title("Accuracy vs CPU latency (marker size ~ params)")
    plt.grid(alpha=0.3)
    savefig("q1_accuracy_vs_latency.png")


def nas_plots():
    p = "results/q2/search_history.csv"
    if not os.path.exists(p):
        return
    h = pd.read_csv(p)
    best = json.load(open("results/q2/best_arch.json")) if os.path.exists("results/q2/best_arch.json") else None
    cons = best["constraints"] if best else None

    plt.figure(figsize=(7, 4.5))
    acc = h[h.status == "accepted"]
    rej = h[h.status == "rejected"]
    plt.scatter(rej.latency_ms, rej.size_mb, marker="x", color=RED, label=f"rejected ({len(rej)})")
    sc = plt.scatter(acc.latency_ms, acc.size_mb, c=acc.val_acc, cmap="viridis", label=f"accepted ({len(acc)})")
    plt.colorbar(sc, label="val top-1 (%)")
    if cons:
        plt.axvline(cons["max_latency_ms"], ls="--", color=GREY)
        plt.axhline(cons["max_size_mb"], ls="--", color=GREY)
    if best:
        plt.scatter(best["hardware"]["latency_ms"], best["hardware"]["size_mb"], s=200, facecolors="none",
                    edgecolors="black", label="best")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("measured CPU latency (ms, log)"); plt.ylabel("model size (MB, log)")
    plt.title("Hardware-aware NAS: candidates vs deployment constraints")
    plt.legend(fontsize=8)
    savefig("q2_nas_constraints.png")

    plt.figure(figsize=(7, 4))
    plt.scatter(acc.trial, acc.val_acc, color=BLUE, s=18, label="accepted trial")
    plt.plot(acc.trial, acc.val_acc.cummax(), color=ORANGE, label="best so far")
    for t in rej.trial:
        plt.axvline(t, color=RED, alpha=0.12)
    plt.xlabel("trial"); plt.ylabel("validation top-1 (%)")
    plt.title("Optuna TPE search history (red lines = rejected)")
    plt.legend(); plt.grid(alpha=0.3)
    savefig("q2_nas_history.png")

    plt.figure(figsize=(7, 4))
    plt.scatter(acc.latency_ms, acc.val_acc, color=BLUE)
    plt.xlabel("measured CPU latency (ms)"); plt.ylabel("validation top-1 (%)")
    plt.title("NAS: accuracy vs latency of feasible candidates"); plt.grid(alpha=0.3)
    savefig("q2_nas_acc_vs_latency.png")


if __name__ == "__main__":
    kd_bar()
    training_curves()
    latency_vs_acc()
    nas_plots()

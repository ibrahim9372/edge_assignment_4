"""
Q2.7 — Load the best architecture discovered by the search and train it FROM SCRATCH
(fresh random initialisation, nothing reused from the proxy search runs) on the full
CIFAR-10 training set, then evaluate on the CIFAR-10 test set.

    python -m q2_nas.train_best --epochs 80
"""
import argparse
import json
import time

import torch

from common.data import GPULoader, load_cifar
from common.train_loop import CECriterion, fit
from common.utils import CSVLogger, count_macs, count_params, save_checkpoint, set_seed, write_json
from q2_nas.generator import build_from_config
from q2_nas.search_space import describe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-json", default="results/q2/best_arch.json")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/nas/nas_best_cifar10.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(args.arch_json) as f:
        cfg = json.load(f)["config"]
    print("Training from scratch:", describe(cfg))
    set_seed(args.seed)
    (xtr, ytr), (xte, yte), classes = load_cifar("cifar10")
    if args.smoke:
        xtr, ytr, args.epochs = xtr[:2048], ytr[:2048], 1
    res = cfg["resolution"]
    tl = GPULoader(xtr, ytr, args.bs, "cifar10", train=True, resolution=res, drop_last=True)
    vl = GPULoader(xte, yte, 1000, "cifar10", train=False, resolution=res)
    model = build_from_config(cfg, 10).cuda().to(memory_format=torch.channels_last)
    print(f"params={count_params(model)/1e6:.3f}M  MMACs={count_macs(model, (1, 3, res, res))/1e6:.1f}")

    t0 = time.time()
    m = fit(model, CECriterion(), tl, vl, args.epochs, args.lr, 5e-4,
            logger=CSVLogger("results/q2/logs/nas_best_train.csv"), tag="nas-best")
    save_checkpoint(args.out, model, "nas", 10, "cifar10", classes, resolution=res, config=cfg,
                    metrics=m, args=vars(args))
    write_json("results/q2/nas_best_final.json", {"config": cfg, "test": m, "epochs": args.epochs,
                                                   "train_minutes": round((time.time() - t0) / 60, 1)})
    print(f"Final NAS model test top1={m['top1']:.2f}")


if __name__ == "__main__":
    main()

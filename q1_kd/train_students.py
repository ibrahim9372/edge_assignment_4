"""
Q1.4 / Q1.5 — Train the three student architectures on CIFAR-100, either

  * baseline  : hard-label cross-entropy only (no teacher involved), or
  * kd        : full distillation framework (hard CE + temperature soft targets +
                feature/hint loss from the frozen ResNet-50 teacher).

Both modes use the *identical* optimiser, schedule, epochs, augmentation and
seed so that any accuracy difference is attributable to distillation.

    python -m q1_kd.train_students --mode baseline --arch all
    python -m q1_kd.train_students --mode kd --arch all --T 4 --alpha 0.3 --beta 1
"""
import argparse
import time

import torch

from common.data import GPULoader, load_cifar
from common.registry import load_checkpoint
from common.train_loop import CECriterion, fit
from common.utils import CSVLogger, count_params, save_checkpoint, set_seed, write_json
from q1_kd.kd import KDCriterion
from q1_kd.models import STUDENTS


def train_one(arch, args, data, teacher=None):
    (xtr, ytr), (xte, yte), classes = data
    set_seed(args.seed)
    train_loader = GPULoader(xtr, ytr, args.bs, "cifar100", train=True, drop_last=True)
    test_loader = GPULoader(xte, yte, 500, "cifar100", train=False)
    model = STUDENTS[arch](num_classes=100).cuda()

    if args.mode == "kd":
        criterion = KDCriterion(teacher, model.feature_channels, teacher.feature_channels,
                                T=args.T, alpha=args.alpha, beta=args.beta,
                                use_soft=not args.no_soft, use_feature=not args.no_feature).cuda()
        extra = criterion.adapter.parameters() if criterion.adapter is not None else None
    else:
        criterion, extra = CECriterion(), None

    print(f"\n=== {args.mode.upper()} | {arch} | params {count_params(model) / 1e6:.2f} M ===")
    t0 = time.time()
    metrics = fit(model, criterion, train_loader, test_loader, args.epochs, args.lr, args.wd,
                  logger=CSVLogger(f"results/q1/logs/{args.mode}_{arch}.csv"), extra_params=extra,
                  tag=f"{args.mode}:{arch}")
    minutes = (time.time() - t0) / 60
    # The adapter is only a training aid; the deployed student is unchanged.
    save_checkpoint(f"checkpoints/{args.mode}/{arch}_cifar100.pt", model, arch, 100, "cifar100",
                    classes, metrics=metrics, args=vars(args))
    return {"arch": arch, "mode": args.mode, **metrics, "train_minutes": round(minutes, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "kd"], required=True)
    ap.add_argument("--arch", default="all", help="mobilenetv2 | squeezenet | efficientnet_lite0 | all")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    # distillation hyper-parameters
    ap.add_argument("--teacher", default="checkpoints/teacher/resnet50_cifar100.pt")
    ap.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    ap.add_argument("--alpha", type=float, default=0.3, help="weight of hard-label CE")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of feature (hint) loss")
    ap.add_argument("--no-soft", action="store_true", help="ablation: disable soft-target loss")
    ap.add_argument("--no-feature", action="store_true", help="ablation: disable feature loss")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    (xtr, ytr), (xte, yte), classes = load_cifar("cifar100")
    if args.smoke:
        xtr, ytr, xte, yte, args.epochs = xtr[:2048], ytr[:2048], xte[:1024], yte[:1024], 1
    data = ((xtr, ytr), (xte, yte), classes)

    teacher = None
    if args.mode == "kd":
        teacher, _ = load_checkpoint(args.teacher, "cuda")

    archs = list(STUDENTS) if args.arch == "all" else [args.arch]
    summary = [train_one(a, args, data, teacher) for a in archs]
    for s in summary:
        print(f"{s['mode']:8s} {s['arch']:20s} top1={s['top1']:.2f} top5={s['top5']:.2f} ({s['train_minutes']} min)")
    if not args.smoke:
        write_json(f"results/q1/{args.mode}_summary.json", summary)


if __name__ == "__main__":
    main()

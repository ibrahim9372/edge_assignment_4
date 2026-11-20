"""
Q1.1 — Train the ResNet-50 teacher on CIFAR-100.

Fine-tuning strategy: start from ImageNet-pretrained ResNet-50 weights (stem conv
re-initialised from the centre 3x3 of the pretrained 7x7 kernel, new 100-way
classifier) and fine-tune *all* layers on CIFAR-100.

    python -m q1_kd.train_teacher --epochs 30
"""
import argparse
import time

import torch

from common.data import GPULoader, load_cifar
from common.registry import load_checkpoint
from common.train_loop import CECriterion, evaluate, fit
from common.utils import CSVLogger, count_params, save_checkpoint, set_seed, write_json
from q1_kd.models import ResNet50, load_imagenet_pretrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch instead")
    ap.add_argument("--out", default="checkpoints/teacher/resnet50_cifar100.pt")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    args = ap.parse_args()

    set_seed(args.seed)
    (xtr, ytr), (xte, yte), classes = load_cifar("cifar100")
    if args.smoke:
        xtr, ytr, xte, yte, args.epochs = xtr[:2048], ytr[:2048], xte[:1024], yte[:1024], 1
    train_loader = GPULoader(xtr, ytr, args.bs, "cifar100", train=True, drop_last=True)
    test_loader = GPULoader(xte, yte, 500, "cifar100", train=False)

    model = ResNet50(num_classes=100)
    if not args.no_pretrained:
        missing, _ = load_imagenet_pretrained(model)
        print(f"Loaded ImageNet weights; newly initialised: {missing}")
    model = model.cuda()
    print(f"ResNet-50 params: {count_params(model) / 1e6:.2f} M")

    t0 = time.time()
    metrics = fit(model, CECriterion(), train_loader, test_loader, args.epochs, args.lr, args.wd,
                  logger=CSVLogger("results/q1/logs/teacher_resnet50.csv"), tag="teacher")
    train_min = (time.time() - t0) / 60

    # Stored in fp16 (~47 MB) so it fits comfortably in a GitHub repo.
    save_checkpoint(args.out, model, "resnet50", 100, "cifar100", classes, metrics=metrics,
                    args=vars(args), half=True)
    # Verify the fp16-stored checkpoint still has the same accuracy.
    reloaded, _ = load_checkpoint(args.out, "cuda")
    m2 = evaluate(reloaded, test_loader)
    print(f"Final teacher top1={metrics['top1']:.2f}  (reloaded fp16 ckpt top1={m2['top1']:.2f})")
    write_json("results/q1/teacher_summary.json",
               {"final": metrics, "reloaded_fp16": m2, "train_minutes": round(train_min, 1), "args": vars(args)})


if __name__ == "__main__":
    main()

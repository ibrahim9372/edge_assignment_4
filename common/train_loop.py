"""
Generic GPU training / evaluation loop shared by teacher, baseline, KD and NAS training.

Design: the *criterion* is a callable  criterion(model, x, y) -> (loss, logits, parts)
so the same loop serves plain cross-entropy training and knowledge distillation
(the KD criterion owns the frozen teacher and the feature adapter).
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class CECriterion(nn.Module):
    """Plain hard-label cross-entropy (baseline training, no distillation)."""

    def forward(self, model, x, y):
        logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        return loss, logits, {"ce": loss.item()}


def cosine_warmup(optimizer, total_steps, warmup_steps):
    """Linear warm-up followed by cosine decay to zero (per-iteration schedule)."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def evaluate(model, loader):
    """Top-1 / top-5 accuracy (%) and mean CE loss."""
    model.eval()
    correct1 = correct5 = total = 0
    loss_sum = 0.0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(1).sum().item()
        total += y.numel()
    return {"top1": 100.0 * correct1 / total, "top5": 100.0 * correct5 / total, "loss": loss_sum / total}


def fit(model, criterion, train_loader, eval_loader, epochs, lr, weight_decay=5e-4,
        warmup_epochs=1, logger=None, extra_params=None, verbose=True, tag=""):
    """Train with SGD(nesterov) + cosine LR + fp16 autocast. Returns final eval metrics.

    extra_params: additional trainable parameters (e.g. KD feature adapter)."""
    params = list(model.parameters()) + (list(extra_params) if extra_params else [])
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    steps = epochs * len(train_loader)
    sched = cosine_warmup(opt, steps, warmup_epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda")
    metrics = {}
    for ep in range(1, epochs + 1):
        model.train()
        if hasattr(criterion, "train"):
            criterion.train()
        t0 = time.time()
        n = 0
        sums = {}
        correct = 0
        for x, y in train_loader:
            with torch.autocast("cuda", dtype=torch.float16):
                loss, logits, parts = criterion(model, x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            bs = y.numel()
            n += bs
            correct += (logits.argmax(1) == y).sum().item()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v * bs
        train_time = time.time() - t0
        metrics = evaluate(model, eval_loader) if eval_loader is not None else {}
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "train_acc": 100.0 * correct / n,
               **{f"train_{k}": v / n for k, v in sums.items()},
               **{f"eval_{k}": v for k, v in metrics.items()}, "epoch_time_s": round(train_time, 2)}
        if logger:
            logger.log(row)
        if verbose:
            parts_s = " ".join(f"{k}={v / n:.3f}" for k, v in sums.items())
            ev = f"eval top1={metrics['top1']:.2f}" if metrics else ""
            print(f"[{tag}] ep {ep:3d}/{epochs} {parts_s} train_acc={100.0 * correct / n:.2f} "
                  f"{ev} ({train_time:.1f}s)", flush=True)
    return metrics

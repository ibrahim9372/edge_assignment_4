"""
Q2.3 / Q2.4 / Q2.6 — Hardware-aware NAS with Bayesian optimisation (Optuna TPE).

For every trial:
  1. TPE samples an architecture configuration from the search space;
  2. the generator builds the network;
  3. the profiler measures real CPU latency, storage size and peak inference memory
     in an isolated subprocess;
  4. if a deployment constraint is violated the trial is REJECTED (pruned) immediately;
  5. otherwise the candidate is trained for a short proxy budget on CIFAR-10
     (45k train split) and validated on a 5k held-out split (the test set is never used
     during the search);
  6. the validation accuracy is the objective that TPE maximises.

Outputs:  results/q2/search_history.csv / .json   (every trial, incl. rejected ones)
          results/q2/best_arch.json                (best feasible architecture)

    python -m q2_nas.search_optuna --trials 50 --epochs 6
"""
import argparse
import json
import time

import optuna
import pandas as pd
import torch

from common.data import GPULoader, load_cifar, train_val_split
from common.train_loop import CECriterion, fit
from common.utils import count_macs, count_params, set_seed, write_json
from q2_nas.generator import build_from_config
from q2_nas.hw_constraints import DeploymentConstraints, check_constraints, profile_candidate
from q2_nas.search_space import describe, sample_with_optuna


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=6, help="proxy training epochs per candidate")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-latency-ms", type=float, default=2.0)
    ap.add_argument("--max-size-mb", type=float, default=1.0)
    ap.add_argument("--max-peak-mem-mb", type=float, default=64.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/q2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cons = DeploymentConstraints(args.max_latency_ms, args.max_size_mb, args.max_peak_mem_mb)
    (xtr, ytr), _, classes = load_cifar("cifar10")
    (xtr, ytr), (xval, yval) = train_val_split(xtr, ytr, 5000, seed=0)
    if args.smoke:
        xtr, ytr, args.trials, args.epochs = xtr[:2048], ytr[:2048], 4, 1

    history, seen = [], {}

    def objective(trial):
        cfg = sample_with_optuna(trial)
        key = json.dumps(cfg, sort_keys=True)
        row = {"trial": trial.number, "arch": describe(cfg), "config": key}
        t0 = time.time()

        hw = profile_candidate(cfg)                         # real CPU measurements
        ok, reasons = check_constraints(hw, cons)
        model = build_from_config(cfg, 10)
        row.update({"params": count_params(model), "mmacs": count_macs(model, (1, 3, cfg["resolution"],
                                                                               cfg["resolution"])) / 1e6,
                    "latency_ms": hw["latency_ms_median"], "size_mb": hw["size_mb"],
                    "peak_mem_mb": hw["peak_mem_mb"]})
        for k, v in row.items():
            if k not in ("config",):
                trial.set_user_attr(k, v)

        if not ok:                                         # hardware-constraint rejection
            row.update({"status": "rejected", "reason": "; ".join(reasons), "val_acc": None,
                        "time_s": round(time.time() - t0, 1)})
            history.append(row)
            _dump(history, args.out)
            print(f"[trial {trial.number:3d}] REJECTED {row['reason']}  :: {row['arch']}", flush=True)
            raise optuna.TrialPruned(row["reason"])

        if key in seen:                                    # identical config already trained
            acc = seen[key]
        else:
            set_seed(args.seed)
            res = cfg["resolution"]
            tl = GPULoader(xtr, ytr, args.bs, "cifar10", train=True, resolution=res, drop_last=True)
            vl = GPULoader(xval, yval, 1000, "cifar10", train=False, resolution=res)
            model = model.cuda().to(memory_format=torch.channels_last)
            m = fit(model, CECriterion(), tl, vl, args.epochs, args.lr, 5e-4, verbose=False)
            acc = m["top1"]
            seen[key] = acc
        row.update({"status": "accepted", "reason": "", "val_acc": acc, "time_s": round(time.time() - t0, 1)})
        history.append(row)
        _dump(history, args.out)
        print(f"[trial {trial.number:3d}] val_acc={acc:.2f}  lat={row['latency_ms']:.2f}ms "
              f"size={row['size_mb']:.3f}MB mem={row['peak_mem_mb']:.1f}MB :: {row['arch']}", flush=True)
        return acc

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="hw_aware_nas")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    best_cfg = json.loads(next(r["config"] for r in history if r["trial"] == best.number))
    summary = {
        "best_trial": best.number, "val_acc": best.value, "config": best_cfg,
        "hardware": {k: best.user_attrs[k] for k in ("latency_ms", "size_mb", "peak_mem_mb", "params", "mmacs")},
        "constraints": cons.to_dict(), "search_epochs_per_trial": args.epochs,
        "n_trials": len(history), "n_rejected": sum(r["status"] == "rejected" for r in history),
        "search_minutes": round((time.time() - t0) / 60, 1), "description": describe(best_cfg),
    }
    write_json(f"{args.out}/best_arch.json", summary)
    print("\nBEST:", json.dumps(summary, indent=2))


def _dump(history, out):
    write_json(f"{out}/search_history.json", history)
    pd.DataFrame(history).to_csv(f"{out}/search_history.csv", index=False)


if __name__ == "__main__":
    main()

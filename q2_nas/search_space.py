"""
Q2.1 — Searchable CNN architecture space.

An architecture is a plain JSON-serialisable dict, e.g.

{
  "resolution": 28,               # variable input resolution
  "stem_channels": 24,
  "activation": "hardswish",      # relu | relu6 | hardswish | silu
  "stages": [                     # 3 stages, stage 2 and 3 halve the resolution
     {"depth": 2, "channels": 32, "kernel": 3, "block": "dwsep"},
     {"depth": 1, "channels": 64, "kernel": 5, "block": "conv"},
     {"depth": 3, "channels": 128, "kernel": 3, "block": "dwsep"}
  ],
  "head_channels": 256,
  "dropout": 0.1
}

Dimensions searched:
  * variable convolutional depth   -> stages[i].depth            {1, 2, 3}
  * variable filters / channels    -> stem / stages[i].channels / head
  * different kernel sizes         -> stages[i].kernel           {3, 5, 7}
  * depthwise-separable blocks     -> stages[i].block            {"conv", "dwsep"}
  * different activations          -> activation                 {relu, relu6, hardswish, silu}
  * variable input resolutions     -> resolution                 {24, 28, 32}

Size of the space: 3 (res) * 3 (stem) * 4 (act) * (3*4*3*2)^3 (stages) * 3 (head) * 2 (dropout)
                   ~= 7.5e7 architectures.
"""

SPACE = {
    "resolution": [24, 28, 32],
    "stem_channels": [16, 24, 32],
    "activation": ["relu", "relu6", "hardswish", "silu"],
    "depth": [1, 2, 3],
    "channels": [[16, 24, 32, 48], [32, 48, 64, 96], [64, 96, 128, 160]],  # per stage
    "kernel": [3, 5, 7],
    "block": ["conv", "dwsep"],
    "head_channels": [128, 256, 384],
    "dropout": [0.0, 0.2],
}
NUM_STAGES = 3


def sample_with_optuna(trial) -> dict:
    """Sample one architecture configuration through an Optuna trial (TPE sampler)."""
    cfg = {
        "resolution": trial.suggest_categorical("resolution", SPACE["resolution"]),
        "stem_channels": trial.suggest_categorical("stem_channels", SPACE["stem_channels"]),
        "activation": trial.suggest_categorical("activation", SPACE["activation"]),
        "stages": [],
        "head_channels": trial.suggest_categorical("head_channels", SPACE["head_channels"]),
        "dropout": trial.suggest_categorical("dropout", SPACE["dropout"]),
    }
    for i in range(NUM_STAGES):
        cfg["stages"].append({
            "depth": trial.suggest_categorical(f"s{i}_depth", SPACE["depth"]),
            "channels": trial.suggest_categorical(f"s{i}_channels", SPACE["channels"][i]),
            "kernel": trial.suggest_categorical(f"s{i}_kernel", SPACE["kernel"]),
            "block": trial.suggest_categorical(f"s{i}_block", SPACE["block"]),
        })
    return cfg


def describe(cfg: dict) -> str:
    """Compact one-line human readable description of a configuration."""
    st = " | ".join(f"{s['block']}{s['kernel']}x{s['kernel']} c{s['channels']} d{s['depth']}" for s in cfg["stages"])
    return f"res{cfg['resolution']} stem{cfg['stem_channels']} {cfg['activation']} [{st}] head{cfg['head_channels']}"

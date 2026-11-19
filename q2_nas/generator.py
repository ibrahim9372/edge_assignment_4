"""
Q2.2 — Automated architecture generator: turns a sampled configuration dict
(see search_space.py) into a concrete nn.Module.

Network template:
    stem:   3x3 conv (stride 1) -> BN -> act
    stage i (i = 0, 1, 2): `depth` blocks; first block of stage 1 and 2 uses stride 2
        block "conv"  : kxk conv -> BN -> act
        block "dwsep" : kxk depthwise conv -> BN -> act -> 1x1 pointwise conv -> BN -> act
        a residual connection is added whenever input/output shapes match
    head:   1x1 conv -> BN -> act -> global avg-pool -> dropout -> linear
"""
import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "relu6": nn.ReLU6, "hardswish": nn.Hardswish, "silu": nn.SiLU}


def conv_bn_act(cin, cout, k, stride, act, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), ACTIVATIONS[act]())


class ConvBlock(nn.Module):
    """Standard (dense) kxk convolution block."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = conv_bn_act(cin, cout, k, stride, act)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.residual else self.body(x)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise kxk conv (per-channel spatial filtering) + pointwise 1x1 conv (channel mixing).
    Cost ~ (k^2 + Cout) / (k^2 * Cout) of a dense conv."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(cin, cin, k, stride, act, groups=cin),
                                  conv_bn_act(cin, cout, 1, 1, act))
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.residual else self.body(x)


BLOCKS = {"conv": ConvBlock, "dwsep": DepthwiseSeparableBlock}


class GeneratedNet(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 10):
        super().__init__()
        self.cfg = cfg
        act = cfg["activation"]
        self.stem = conv_bn_act(3, cfg["stem_channels"], 3, 1, act)
        layers, cin = [], cfg["stem_channels"]
        for i, st in enumerate(cfg["stages"]):
            for d in range(st["depth"]):
                stride = 2 if (i > 0 and d == 0) else 1
                layers.append(BLOCKS[st["block"]](cin, st["channels"], st["kernel"], stride, act))
                cin = st["channels"]
        self.features = nn.Sequential(*layers)
        self.head = conv_bn_act(cin, cfg["head_channels"], 1, 1, act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
        self.fc = nn.Linear(cfg["head_channels"], num_classes)

    def forward(self, x):
        x = self.head(self.features(self.stem(x)))
        return self.fc(self.dropout(torch.flatten(self.pool(x), 1)))


def build_from_config(cfg: dict, num_classes: int = 10) -> nn.Module:
    return GeneratedNet(cfg, num_classes)

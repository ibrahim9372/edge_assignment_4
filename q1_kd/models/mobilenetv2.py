"""
MobileNetV2 (Sandler et al., 2018) student, adapted to 32x32 CIFAR inputs.

Building block = inverted residual:  1x1 expand (xt) -> 3x3 depthwise -> 1x1 linear projection,
with an identity shortcut when stride == 1 and in/out channels match.

CIFAR adaptation: stem stride 2 -> 1 and the first stride-2 stage (24 ch) -> stride 1,
so the network downsamples 32 -> 4 instead of 224 -> 7.
Spatial sizes: 16/24 ch @32x32 | 32 ch @16x16 | 64/96 ch @8x8 | 160/320 ch @4x4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin, cout, k=3, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand, kernel=3):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers.append(conv_bn_act(cin, hidden, 1))                          # expand
        layers.append(conv_bn_act(hidden, hidden, kernel, stride, groups=hidden))  # depthwise
        layers.append(conv_bn_act(hidden, cout, 1, act=False))                 # linear bottleneck
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class MobileNetV2(nn.Module):
    # (expansion t, out channels c, repeats n, stride s) — strides adapted for CIFAR
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    FEAT_STAGE = 4  # output of the 96-ch stage (8x8) is the KD hint layer

    def __init__(self, num_classes=100, dropout=0.2):
        super().__init__()
        self.stem = conv_bn_act(3, 32, 3, 1)
        cin = 32
        stages = []
        for t, c, n, s in self.CFG:
            blocks = []
            for i in range(n):
                blocks.append(InvertedResidual(cin, c, s if i == 0 else 1, t))
                cin = c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.feature_channels = self.CFG[self.FEAT_STAGE][1]
        self.head = conv_bn_act(cin, 1280, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        feat = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == self.FEAT_STAGE:
                feat = x
        x = self.head(x)
        logits = self.fc(self.dropout(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)))
        return (logits, feat) if return_features else logits

"""
EfficientNet-Lite0 student, adapted to 32x32 CIFAR inputs.

EfficientNet-Lite is the edge/quantisation-friendly variant of EfficientNet-B0:
  * Squeeze-and-Excitation blocks removed;
  * Swish replaced by ReLU6;
  * stem (32 ch) and head (1280 ch) are fixed and not scaled.
Building block = MBConv (inverted residual with 3x3 or 5x5 depthwise conv).
Lite0 uses width = depth = 1.0 (the B0 baseline layout).

CIFAR adaptation: stem stride 2 -> 1 and the first stride-2 stage (24 ch) -> stride 1.
Spatial sizes: 16/24 ch @32x32 | 40 ch @16x16 | 80/112 ch @8x8 | 192/320 ch @4x4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilenetv2 import InvertedResidual, conv_bn_act


class EfficientNetLite0(nn.Module):
    # (expansion, kernel, stride, out channels, repeats) — strides adapted for CIFAR
    CFG = [(1, 3, 1, 16, 1), (6, 3, 1, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
           (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]
    FEAT_STAGE = 4  # 112-ch stage output (8x8) is the KD hint layer

    def __init__(self, num_classes=100, dropout=0.2):
        super().__init__()
        self.stem = conv_bn_act(3, 32, 3, 1)
        cin = 32
        stages = []
        for t, k, s, c, n in self.CFG:
            blocks = []
            for i in range(n):
                # MBConv without SE, ReLU6 activation == Lite variant
                blocks.append(InvertedResidual(cin, c, s if i == 0 else 1, t, kernel=k))
                cin = c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.feature_channels = self.CFG[self.FEAT_STAGE][3]
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

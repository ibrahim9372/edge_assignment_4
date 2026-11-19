"""
SqueezeNet (Iandola et al., 2016; v1.1 layout) student, adapted to 32x32 CIFAR inputs.

Fire module: 1x1 "squeeze" conv (few channels) followed by parallel 1x1 and 3x3
"expand" convs whose outputs are concatenated. Classification is done by a 1x1
conv to num_classes followed by global average pooling (no fully-connected layer).

Modifications for CIFAR / stable training from scratch:
  * stem is a 3x3 stride-1 conv (instead of 3x3/2 on 224x224);
  * BatchNorm after every conv (original has none) — allows LR 0.1 + SGD;
  * no ReLU on the final class-score map (it would clip negative logits).
Spatial sizes: stem 32x32 -> pool 16x16 (fire2,3) -> pool 8x8 (fire4,5) -> pool 4x4 (fire6-9).
"""
import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, k, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Fire(nn.Module):
    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = conv_bn_relu(cin, squeeze, 1)
        self.expand1 = conv_bn_relu(squeeze, e1, 1)
        self.expand3 = conv_bn_relu(squeeze, e3, 3)

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1(s), self.expand3(s)], dim=1)


class SqueezeNet(nn.Module):
    feature_channels = 256  # fire5 output (8x8) used as the KD hint layer

    def __init__(self, num_classes=100, dropout=0.5):
        super().__init__()
        self.stem = conv_bn_relu(3, 64, 3, 1)
        self.stage1 = nn.Sequential(nn.MaxPool2d(3, 2, 1), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64))
        self.stage2 = nn.Sequential(nn.MaxPool2d(3, 2, 1), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128))
        self.stage3 = nn.Sequential(nn.MaxPool2d(3, 2, 1), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                                    Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1),
                                        nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x, return_features=False):
        x = self.stage1(self.stem(x))
        feat = self.stage2(x)
        logits = self.classifier(self.stage3(feat))
        return (logits, feat) if return_features else logits

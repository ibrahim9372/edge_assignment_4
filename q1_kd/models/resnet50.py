"""
ResNet-50 teacher (bottleneck residual network, layers [3, 4, 6, 3]).

CIFAR adaptation: the ImageNet stem (7x7 stride-2 conv + max-pool, i.e. 4x
downsampling) would shrink a 32x32 image to 8x8 before the first residual
stage, so it is replaced with a 3x3 stride-1 conv and no max-pool.
Spatial sizes for a 32x32 input:  layer1 32x32 | layer2 16x16 | layer3 8x8 | layer4 4x4.

Parameter names deliberately mirror torchvision's ResNet so ImageNet
pre-trained weights can be loaded for fine-tuning (all layers except the stem
conv, which changes shape, and the final classifier).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet50(nn.Module):
    feature_channels = 1024  # channels of the layer3 output used as the KD "hint"

    def __init__(self, num_classes=100):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # CIFAR stem
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)
        self.fc = nn.Linear(512 * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, planes, blocks, stride):
        down = None
        if stride != 1 or self.inplanes != planes * 4:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                                 nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, down)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        feat = self.layer3(x)               # (B, 1024, 8, 8) intermediate representation
        x = self.layer4(feat)
        logits = self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
        return (logits, feat) if return_features else logits


def load_imagenet_pretrained(model: ResNet50):
    """Initialise from torchvision's ImageNet ResNet-50 weights (fine-tuning start point).

    * fc.* is skipped (1000 -> num_classes).
    * conv1 (7x7) is replaced by its centre 3x3 crop to initialise our 3x3 CIFAR stem."""
    import torchvision
    sd = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    sd["conv1.weight"] = sd["conv1.weight"][:, :, 2:5, 2:5].clone()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected

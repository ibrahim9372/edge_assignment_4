from .resnet50 import ResNet50, load_imagenet_pretrained
from .mobilenetv2 import MobileNetV2
from .squeezenet import SqueezeNet
from .efficientnet_lite0 import EfficientNetLite0

STUDENTS = {
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}

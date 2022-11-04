# from models.resnet import ResNet50
# from models.small import LeNet, FC1024, BNNet
from models.gemresnet import GEMResNet18
from models.gemresnet_csi import GEMResNet18CSI
from models.alexnet import Alexnet
from models.alexnet_csi import AlexnetCSI

__all__ = [
    # "LeNet",
    # "FC1024",
    # "BNNet",
    # "ResNet50",
    "GEMResNet18",
    "GEMResNet18CSI",
    "Alexnet",
    "AlexnetCSI"
]
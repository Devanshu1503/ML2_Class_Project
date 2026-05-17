import torch.nn as nn
import torchvision.models as models


def build_mobilenet(num_classes: int, freeze_backbone: bool = True):
    """
    MobileNetV3-Large pretrained on ImageNet-1K.

    Architecture highlights:
      - Inverted residual blocks with depthwise separable convolutions
        (splits a standard conv into a spatial depthwise filter + a 1×1
        pointwise mixer, cutting parameters ~8–9× vs a standard conv)
      - Squeeze-and-Excitation (SE) channel attention in every block
        (globally averages each feature map, learns which channels matter)
      - Hard-Swish activation: a fast piecewise approximation of Swish
      - Built-in Dropout(0.2) before the final classifier
      - Only 5.4M parameters — ideal for small datasets

    Classifier structure (torchvision):
      [0] Linear(960 → 1280)
      [1] Hardswish
      [2] Dropout(0.2)        ← built-in regularization
      [3] Linear(1280 → 1000) ← replaced with Linear(1280 → num_classes)
    """
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True

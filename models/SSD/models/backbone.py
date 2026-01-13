import torch.nn as nn
from torchvision.models import vgg16, mobilenet_v2

class Backbone(nn.Module):
    def __init__(self, name="vgg"):
        super().__init__()

        if name == "vgg":
            # vgg = vgg16(pretrained=False)
            vgg = vgg16(weights=None).features
            self.features = vgg.features
            self.out_channels = [512, 1024]

        elif name == "mobilenet":
            mobilenet = mobilenet_v2(pretrained=False)
            self.features = mobilenet.features
            self.out_channels = [96, 1280]

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [22, len(self.features)-1]:
                features.append(x)
        return features

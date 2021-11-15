import torch
import torch.nn as nn
import numpy as np

def UNet(in_channels=1, out_channels=1, init_features=32, pretrained=False):
    UNet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=in_channels, out_channels=out_channels, init_features=init_features, pretrained=pretrained)
    return UNet


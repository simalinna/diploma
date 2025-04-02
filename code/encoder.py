import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet50


class MyResnet(nn.Module):
    def __init__(self, encoder_dim):
        super(MyResnet, self).__init__()

        self.encoder = resnet18(num_classes=encoder_dim)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1)
        self.encoder.maxpool = nn.Identity()

        # self.encoder = resnet50(num_classes=encoder_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        return x
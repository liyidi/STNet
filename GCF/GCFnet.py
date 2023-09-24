from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from tools import ops

# @torch.enable_grad()
class audioNet(nn.Module):
    def __init__(self):
        super(audioNet, self).__init__()
        self.net_head = AlexNetV1_au()
        ops.init_weights(self.net_head)
        self.predNet = GCFpredictor()
        ops.init_weights(self.predNet)
    def forward(self, auFr):
        Fau = self.net_head(auFr).permute(0, 2, 3, 1)  # [b,c,h,w]-->[b,h,w,c]
        out_fc = self.predNet(Fau)
        out_pred = torch.mean(out_fc,axis = -1)
        return out_pred


class _AlexNet_au(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class AlexNetV1_au(_AlexNet_au):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1_au, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 6, 1, groups=2))


class GCFpredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
        nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
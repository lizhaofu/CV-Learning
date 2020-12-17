#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/17 16:45
# @File    : residualnet_nodel.py
# @Software: PyCharm
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.cov1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cov2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.cov1(x))
        y = self.cov2(y)
        return F.relu(x + y)


class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()
        self.cov1 = nn.Conv2d(1, 16, kernel_size=5)
        self.rblocl1 = ResidualBlock(16)

        self.cov2 = nn.Conv2d(16, 32, kernel_size=5)
        self.rblocl2 = ResidualBlock(32)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.mp(F.relu(self.cov1(x)))
        x = self.rblocl1(x)
        x = self.mp(F.relu(self.cov2(x)))
        x = self.rblocl2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(8, 1, 28, 28)

    model = ResidualNet()
    print(model)

    output = model(x)

    print(output.shape)
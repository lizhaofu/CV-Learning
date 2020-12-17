#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/5 15:46
# @File    : tesy.py
# @Software: PyCharm
"""
import torch
import torchvision.models as models
from tensorboard import summary
from torch.utils.tensorboard import SummaryWriter

model = models.resnet50()
dummy_input = torch.randn(1, 3, 224, 224)
with SummaryWriter(comment='Resnet50') as w:
    w.add_graph(model, ((dummy_input,)))

import torch
import torchvision.models as models

import time


model = models.resnet50().cuda()
summary(model, input_size=(3, 224, 244))
# inference time
inputs = torch.randn(1, 3, 224, 224).cuda()
end = time.time()
y = model(inputs)
print("inference time:{}".format(time.time() - end))
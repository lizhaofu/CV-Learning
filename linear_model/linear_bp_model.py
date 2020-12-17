#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/16 16:59
# @File    : linear_bp_model.py
# @Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]

# initial guss of weight

w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('Predict (before training)', 5, forward(5).item())


# training
epoch_list = []
loss_list = []

for epoch in range(100):
    epoch_list.append(epoch)
    l_sum = 0
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        loss_val.backward()
        print('\tgrad: ', x, y, w.grad.item())
        l_sum += loss_val.item()
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    loss_list.append(l_sum / 4)
    print('epoch: ', epoch, 'w=', w, 'loss=', l_sum / 4)

# prediction
print('Predict (after training)', 5, forward(5).item())



#  draw loss
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss_bp.jpg')
plt.show()
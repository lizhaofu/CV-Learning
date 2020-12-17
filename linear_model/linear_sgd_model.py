#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/16 16:39
# @File    : linear_sgd_model.py
# @Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]

# initial guss of weight

w = 1.0


def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

print('Predict (before training)', 5, forward(5))


# training
epoch_list = []
loss_list = []

for epoch in range(100):
    epoch_list.append(epoch)
    l_sum = 0
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        l_sum += loss_val
        grad_val = gradient(x,y)
        print('\tgrad: ', x, y, grad_val)
        w -= 0.01 * grad_val
    loss_list.append(l_sum / 4)
    print('epoch: ', epoch, 'w=', w, 'loss=', l_sum / 4)

# prediction
print('Predict (after training)', 5, forward(5))



#  draw loss
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss_sgd.jpg')
plt.show()


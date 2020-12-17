#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/16 16:10
# @File    : linear_gradient_descent_model.py
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

def loss(xs, ys):
    loss = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        loss = (y_pred - y) ** 2
    return loss / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

print('Predict (before training)', 5, forward(5))


# training
epoch_list = []
loss_list = []
for epoch in range(100):
    epoch_list.append(epoch)
    loss_val = loss(x_data, y_data)
    loss_list.append(loss_val)
    grad_val = gradient(x_data,y_data)
    w -= 0.01 * grad_val
    print('epoch: ', epoch, 'w=', w, 'loss=', loss_val)

# prediction
print('Predict (after training)', 5, forward(5))



#  draw loss
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss.jpg')
plt.show()


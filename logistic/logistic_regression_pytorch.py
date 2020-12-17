#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/16 19:59
# @File    : logistic_regression_pytorch.py
# @Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

x_data = [[1.0], [2.0], [3.0], [4.0]]
y_data = [[0], [0], [1], [1]]

x_data = torch.Tensor(x_data)
y_data = torch.Tensor(y_data)


# print(x_data, '\n', y_data)

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear(x)
        x = F.sigmoid(x)
        return x


model = LogisticRegressionModel()

# print(model)

loss_function = nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_list = []

for epoch in range(10000):
    if torch.cuda.is_available():
        x_data, y_data = x_data.cuda(), y_data.cuda()
        model = model.cuda()
    y_pred = model(x_data)
    loss = loss_function(y_pred, y_data)
    print('epoch: ', epoch, 'loss=', loss.item())
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# prediction
x_test = torch.Tensor([[5.0]])
if torch.cuda.is_available():
    x_test = x_test.cuda()
y_test = model(x_test)
print('y_pred = ', y_test.data)

# print('w = ', model.linear.weight.item())
# print('b = ', model.linear.bias.item())


#  draw loss
plt.plot(range(10000), loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss_pytorch.jpg')
plt.show()

x = np.linspace(1, 10, 200)
x_t = torch.Tensor(x).view((200, 1)).cuda()
y_t = model(x_t)
y = y_t.data.cpu().numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('hours')
plt.ylabel('probability of pass')
plt.grid()
plt.show()

#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/16 19:59
# @File    : multi_dim_logistic_regression_pytorch.py
# @Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

data = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(data[:, :-1])
y_data = torch.from_numpy(data[:, [-1]])
print(x_data.size())
print(y_data.size())

# print(x_data, '\n', y_data)

class MultiDimModel(nn.Module):
    def __init__(self):
        super(MultiDimModel, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


model = MultiDimModel()

print(model)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_list = []

if torch.cuda.is_available():
    x_data, y_data = x_data.cuda(), y_data.cuda()
    model = model.cuda()

for epoch in range(10000):
    # Forward

    y_pred = model(x_data)
    loss = loss_function(y_pred, y_data)
    print('epoch: ', epoch, 'loss=', loss.item())
    loss_list.append(loss.item())

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

# prediction

x_test = x_data[1]
print(x_test)
print(y_data[1])
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

# x = np.linspace(1, 10, 200)
# x_t = torch.Tensor(x).view((200, 1)).cuda()
# y_t = model(x_t)
# y = y_t.data.cpu().numpy()
# plt.plot(x, y)
# plt.plot([0, 10], [0.5, 0.5], c='r')
# plt.xlabel('hours')
# plt.ylabel('probability of pass')
# plt.grid()
# plt.show()

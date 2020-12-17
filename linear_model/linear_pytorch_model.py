#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/16 17:15
# @File    : linear_pytorch_model.py
# @Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

x_data = [[1.0], [2.0], [3.0], [4.0]]
y_data = [[2.0], [4.0], [6.0], [8.0]]

x_data = torch.Tensor(x_data)
y_data = torch.Tensor(y_data)

# print(x_data, '\n', y_data)

class Linear_model(nn.Module):
    def __init__(self):
        super(Linear_model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


model = Linear_model()

# print(model)

loss_function = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


epoch_list = []
loss_list = []

for epoch in range(200):
    epoch_list.append(epoch)
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

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())



#  draw loss
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss_pytorch.jpg')
plt.show()
#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/3 11:12
# @File    : mlp-pytorch.py
# @Software: PyCharm
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
import torch.nn.functional as F
import time

start = time.time()
device = torch.device('cpu')

n_feature, n_hidden, n_output = 4, 2000, 3

dataset = datasets.load_iris()
X_train = dataset['data']
y = dataset['target']

# print(X_train)
# print(y)

X_train = torch.FloatTensor(X_train)
y = torch.LongTensor(y)


# print(X_train)
# print(y)


class MLPNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLPNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x


net = MLPNet(n_feature, n_hidden, n_output)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_function = torch.nn.CrossEntropyLoss().to(device)

X_train, y = X_train.to(device), y.to(device)
net = net.to(device)

for i in range(10000):
    out = net(X_train)
    loss = loss_function(out, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

# out = net(X_train)
# prediction = torch.max(out, 1)[1]
# pred_y = prediction.cpu().numpy()
# target_y = y.data.cpu().numpy()
# pred_y = prediction.numpy()
# target_y = y.data.numpy()

# print(pred_y)
# print(target_y)

end = time.time()
print('Running time: %s Seconds' % (end - start))

start = time.time()
for i in range(10000):
    out = net(X_train)
    loss = loss_function(out, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

# out = net(X_train)
# prediction = torch.max(out, 1)[1]
# pred_y = prediction.cpu().numpy()
# target_y = y.data.cpu().numpy()
end = time.time()
print('Running time: %s Seconds' % (end - start))


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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


filepath = 'diabetes.csv.gz'

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.from_numpy(data[:, [-1]])

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

dataset = DiabetesDataset(filepath)

train_loader = DataLoader(dataset=dataset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=6)


# print(x_data, '\n', y_data)

class MultiDimModel(nn.Module):
    def __init__(self):
        super(MultiDimModel, self).__init__()
        self.linear1 = nn.Linear(8, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


model = MultiDimModel()

print(model)

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_list = []

if torch.cuda.is_available():
    model = model.cuda()


for epoch in range(10000):
    loss_tmp = 0

    for i, data in enumerate(train_loader, 1):
        # 1. prepare data
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # 2. Forward
        y_pred = model(inputs)
        loss = loss_function(y_pred, labels)
        if epoch % 500 == 0:
            print('epoch: ', epoch, 'batch: ', i, 'loss=', loss.item())
        loss_tmp = loss.item()


        # 3. Backward
        optimizer.zero_grad()
        loss.backward()

        # 4. Update
        optimizer.step()
    loss_list.append(loss_tmp)

#  draw loss
plt.plot(range(10000), loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss_pytorch.jpg')
plt.show()


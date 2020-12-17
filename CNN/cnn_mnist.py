#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/17 14:25
# @File    : cnn_mnist.py
# @Software: PyCharm
"""
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

'''
# CNN Test
# CNN layer
in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size, in_channels, width, height)

conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

out_put = conv_layer(input)

print(input.size())
print(out_put.shape)
print(conv_layer.weight.shape)

# CNN layer with padding

input_1 = [3, 4, 6, 5, 7,
           2, 4, 6, 8, 2,
           1, 6, 7, 8, 4,
           9, 7, 4, 6, 2,
           3, 7, 5, 4, 1]

input_1 = torch.Tensor(input_1).view(1, 1, 5, 5)

print('input_1', input_1)

# conv_layer_1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
conv_layer_1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
conv_layer_1.weight.data = kernel.data

output = conv_layer_1(input_1)
print(output)


input = [3,4,6,5,
         2,4,6,8,
         1,6,7,8,
         9,7,4,6]
input = torch.Tensor(input).view(1, 1, 4, 4)

maxpooling_layer_1 = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer_1(input) 
print(output)
'''


BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=BATCH_SIZE,
                          num_workers=8,
                          pin_memory=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=BATCH_SIZE,
                         num_workers=8,
                         pin_memory=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)


    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # Flatten
        x = self.fc(x)
        return x


# model = Net()
#
# print(model)
#
# x_data = torch.randn(1, 1, 28, 28)
# print(model(x_data).view(1, -1).shape)



def train(model, device, train_loader, optimizer, loss_function, epoch):
    model.train()
    # running_loss =0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        output = model(data)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        # running_loss += loss.item
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # running_loss = 0.0


def test(model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))


def main():
    cudnn.benchmark = True
    torch.manual_seed(1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    model = Net().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, loss_function, epoch)

        test(model, device)

if __name__ == '__main__':
    main()
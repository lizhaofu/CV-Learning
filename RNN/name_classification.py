#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-
"""
# @Company : 华中科技大学
# @version : V1.0
# @Author  : lizhaofu
# @contact : lizhaofu0215@163.com  2018--2022
# @Time    : 2020/12/17 21:44
# @File    : name_classification.py
# @Software: PyCharm
"""
import csv
import gzip
import math
import time

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np

# parameters

HIDDEN_SIZE = 256
EMBEDDING_SIZE = 256
BATCH_SIZE = 256
N_LAYER = 1
N_EPOCHS = 20
N_CHARS = 128
USE_GPU = True

class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'names_train.csv.gz' if is_train_set else 'names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))--force
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, item):
        return self.names[item], self.country_dict[self.countries[item]]

    def __len__(self):
        return  self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num


trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

N_COUNTRY = trainset.getCountriesNum()


class RNNClassfier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassfier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.bn = nn.BatchNorm1d(hidden_size * self.n_directions)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)

        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        hidden_cat = self.bn(hidden_cat)
        fc_output = self.fc(hidden_cat)
        return fc_output

def create_tensor(tensor):
    if USE_GPU:
        device = torch.device('cuda')
        tensor = tensor.to(device)
    return tensor

def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()

    # make tensor of name, batchsize * seqlen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), seq_lengths, create_tensor(countries)


def train_model(epoch, trainset, trainloader, classifier, loss_function, optimizer, start):
    total_loss = 0
    total = len(trainset)
    classifier.train()
    for idx, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)

        # print(inputs.shape, target.shape)
        # print(target)
        #
        # _, predicted = torch.max(output.data, dim=1)
        # print(output.shape)
        # print(output)
        # pred = output.max(dim=1, keepdim=True)[1]
        # print("output", output.max(dim=1)[1])
        # print(output.max(dim=1)[1].shape)
        # print(predicted)
        # print(predicted.shape)
        # print(target.view_as(pred))
        # print(target.view_as(pred).shape)
        # print('*' * 30)

        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if idx % 20 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{idx * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (idx * len(inputs))}')
    return total_loss  / total

def test_model(testset, testloader, loss_function, classifier):

    correct = 0
    total = len(testset)
    total_loss = 0
    classifier.eval()
    print("evaluating trained model ...")

    with torch.no_grad():
        for idx, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            # print(output)
            loss = loss_function(output, target)
            pred = output.max(dim=1, keepdim=True)[1]
            # print(pred.shape)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # correct += (pred == target).sum().item()
            total_loss += loss.item()

        percent = '%.2f' % (100 * correct / total)

        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total, total_loss  / total

def main():
    cudnn.benchmark = True
    torch.manual_seed(1)
    classifier = RNNClassfier(N_CHARS, EMBEDDING_SIZE, HIDDEN_SIZE, N_COUNTRY, N_LAYER, bidirectional=False)
    if USE_GPU:
        device = torch.device('cuda')
        classifier.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(classifier.parameters(), lr=0.0001, weight_decay=1e-5)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    loss_list = []
    evl_loss_list = []
    for epoch in range(1, N_EPOCHS + 1):
        loss = train_model(epoch, trainset, trainloader, classifier, loss_function, optimizer, start)
        loss_list.append(loss)
        acc, evl_loss = test_model(testset, testloader, loss_function, classifier)
        acc_list.append(acc)
        evl_loss_list.append(evl_loss)

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    loss_list = np.array(loss_list)
    evl_loss_list = np.array(evl_loss_list)
    plt.plot(epoch, loss_list)
    plt.plot(epoch, evl_loss_list, color='red', linewidth=1, linestyle='--')
    # plt.plot(epoch, acc_list, color='green', linewidth=1, linestyle='dotted')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig('loss_pytorch.jpg')
    plt.show()

if __name__ == '__main__':
    main()




    # classifier = RNNClassfier(N_CHARS, EMBEDDING_SIZE, HIDDEN_SIZE, N_COUNTRY, N_LAYER, bidirectional=False)
    # print(classifier)






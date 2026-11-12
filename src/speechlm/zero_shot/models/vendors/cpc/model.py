# Trimmed to what the pretrained checkpoint uses, from
# https://github.com/JAMJU/Sel_supervised_models_perception_biases/blob/main/CPC/cpc/model.py
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelNorm(nn.Module):
    def __init__(self, numFeatures, epsilon=1e-05):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.ones(1, numFeatures, 1))
        self.bias = nn.parameter.Parameter(torch.zeros(1, numFeatures, 1))
        self.epsilon = epsilon

    def forward(self, x):
        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean) * torch.rsqrt(cumVar + self.epsilon)
        return x * self.weight + self.bias


class CPCEncoder(nn.Module):
    def __init__(self, sizeHidden=256):
        super().__init__()
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = ChannelNorm(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = ChannelNorm(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm2 = ChannelNorm(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = ChannelNorm(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = ChannelNorm(sizeHidden)

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        return F.relu(self.batchNorm4(self.conv4(x)))


class CPCAR(nn.Module):
    def __init__(self, dimEncoded=256, dimOutput=256):
        super().__init__()
        self.baseNet = nn.LSTM(dimEncoded, dimOutput, num_layers=1, batch_first=True)


class CPCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gEncoder = CPCEncoder()
        self.gAR = CPCAR()

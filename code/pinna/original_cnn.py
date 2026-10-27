"""Original PinnaResNet class definitions from the supplied notebook, cell 20.

Architecture and forward methods preserved; notebook training is not executed.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=(7, 3), pool=True):
        super().__init__()
        kf, kt = kernel_size
        pad = (kf // 2, kt // 2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        z = self.act(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        z = self.act(z + identity)
        return self.pool(z)

class SqueezeExcite(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=2)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)

class SpecAugment(nn.Module):

    def __init__(self, freq_mask=8, time_mask=20, n_freq=2, n_time=2):
        super().__init__()
        self.freq_mask, self.time_mask = (freq_mask, time_mask)
        self.n_freq, self.n_time = (n_freq, n_time)

    def forward(self, x):
        if not self.training:
            return x
        x = x.clone()
        _, _, F, T = x.shape
        for _ in range(self.n_freq):
            f = random.randint(0, self.freq_mask)
            f0 = random.randint(0, max(0, F - f))
            x[:, :, f0:f0 + f, :] = 0
        for _ in range(self.n_time):
            t = random.randint(0, self.time_mask)
            t0 = random.randint(0, max(0, T - t))
            x[:, :, :, t0:t0 + t] = 0
        return x

class PinnaResNet(nn.Module):

    def __init__(self, in_ch=4, dropout=0.3):
        super().__init__()
        self.spec_aug = SpecAugment(freq_mask=8, time_mask=20)
        self.block1 = ResBlock(in_ch, 32, kernel_size=(7, 3), pool=True)
        self.block2 = ResBlock(32, 64, kernel_size=(7, 3), pool=True)
        self.block3 = ResBlock(64, 128, kernel_size=(5, 3), pool=True)
        self.block4 = ResBlock(128, 256, kernel_size=(3, 3), pool=False)
        self.freq_conv = nn.Conv1d(256, 256, kernel_size=5, padding=2, bias=False)
        self.freq_bn = nn.BatchNorm1d(256)
        self.se = SqueezeExcite(256, reduction=8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(256, 3)

    def forward(self, x):
        x = self.spec_aug(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean(dim=-1)
        x = F.relu(self.freq_bn(self.freq_conv(x)), inplace=True)
        x = self.se(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        v = self.head(x)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-08)
        return v

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class DnCNN(nn.Module):
    def __init__(self, depth=20, n_channels=64, image_channels=1, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y+out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        print('init weight successfully')
class DenseCNN(nn.Module):
    def __init__(self, depth=10, n_channels=64, image_channels=1, kernel_size=3):
        super(DenseCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers1=[]
        layers2=[]
        layers3=[]
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        layers1.append(nn.Conv2d(in_channels=image_channels+1, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers1.append(nn.LeakyReLU(inplace=True))
        for _ in range(depth-2):
            layers1.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers1.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers1.append(nn.LeakyReLU(inplace=True))
        layers1.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn1 = nn.Sequential(*layers1)
        layers2.append(nn.Conv2d(in_channels=image_channels+2, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers2.append(nn.LeakyReLU(inplace=True))
        for _ in range(depth-2):
            layers2.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers2.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers2.append(nn.LeakyReLU(inplace=True))
        layers2.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn2 = nn.Sequential(*layers2)
        layers3.append(nn.Conv2d(in_channels=image_channels+3, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers3.append(nn.LeakyReLU(inplace=True))
        for _ in range(depth-2):
            layers3.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers3.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers3.append(nn.LeakyReLU(inplace=True))
        layers3.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn3 = nn.Sequential(*layers3)
        self._initialize_weights()

    def forward(self, x):
        out1 = self.dncnn(x)
        out  = torch.cat([out1,x],1)
        out2 = self.dncnn1(out)
        out  = torch.cat([out2,out1,x],1)
        out3 = self.dncnn2(out)
        out  = torch.cat([out3,out2,out1,x],1)
        out  = self.dncnn3(out)
        return out+x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        print('init weight successfully')
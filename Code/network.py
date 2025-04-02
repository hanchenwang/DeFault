#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
from transformer import SeisT
#from mlp_mixer_pytorch import MLPMixer
from math import ceil
from collections import OrderedDict

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
              .replace('Conv2DwithBN_Tanh', 'layers')
              .replace('Deconv2DwithBN', 'layers')
              .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=0.2, dropout=False):
        super(Conv2DwithBN,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class Conv2DwithBNUQ(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=0.2, dropout=True):
        super(Conv2DwithBNUQ,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(F.dropout2d(p=0.5,training=True))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)


class LinearwithBN(nn.Module):
    def __init__(self, in_fea, out_fea,
                bn=True, relu_slop=0.2, dropout=True):
        super(LinearwithBN,self).__init__()
        layers = [nn.Linear(in_fea, out_fea)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        self.LinearwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.LinearwithBN(x)

class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(Conv2DwithBN_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)


class Deconv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, dropout=False):
        super(Deconv2DwithBN, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        self.Deconv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Deconv2DwithBN(x)

class Deconv3DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, dropout=False):
        super(Deconv3DwithBN, self).__init__()
        layers = [nn.ConvTranspose3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        layers.append(nn.BatchNorm3d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout3d(0.5))
        self.Deconv3DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Deconv3DwithBN(x)


class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)


class FCN4_No_BN(nn.Module):
    def __init__(self, ns=5, upsample_mode='nearest'):
        super(FCN4_No_BN, self).__init__()
        ch = 2 ** 5
        layers = [nn.Conv2d(ns, ch, (7, 1), (2, 1), (3, 0)), nn.LeakyReLU(0.2, inplace=True)]
        for i in range(12):
            kernel_size, padding = [3, 3], [1, 1]
            stride = [2, 2] if i % 2 == 0 else [1, 1]
            if i < 6:
                kernel_size[1], stride[1], padding[1] = 1, 1, 0
            if i % 4 == 0:
                layers.append(nn.Conv2d(ch, ch * 2, tuple(kernel_size), tuple(stride), tuple(padding)))
                ch *= 2
            else:
                layers.append(nn.Conv2d(ch, ch, tuple(kernel_size), tuple(stride), tuple(padding)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.extend([nn.Conv2d(ch, ch*2, (8, 9), 1, 0), nn.LeakyReLU(0.2, inplace=True)])
        ch *= 2 
        self.encoder = nn.Sequential(*layers)

        layers = []
        for i in range(10):
            if i % 2 == 0:
                layers.append(nn.Upsample(scale_factor=5 if i == 0 else 2, mode=upsample_mode))
            out_ch = ch if i == 0 or i % 2 == 1 else ch // 2
            layers.extend([nn.Conv2d(ch, out_ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)])
            ch = out_ch
        self.decoder = nn.Sequential(*layers)

        self.output = nn.Sequential(nn.Conv2d(ch, 1, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        return self.output(x)


class FCN4_BN(nn.Module):
    def __init__(self, ns=5, upsample_mode='nearest'):
        super(FCN4_BN, self).__init__()
        ch = 2 ** 5
        layers = [nn.Conv2d(ns, ch, (7, 1), (2, 1), (3, 0)), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True)]
        for i in range(12):
            kernel_size, padding = [3, 3], [1, 1]
            stride = [2, 2] if i % 2 == 0 else [1, 1]
            if i < 6:
                kernel_size[1], stride[1], padding[1] = 1, 1, 0
            if i % 4 == 0:
                layers.append(nn.Conv2d(ch, ch * 2, tuple(kernel_size), tuple(stride), tuple(padding)))
                ch *= 2
            else:
                layers.append(nn.Conv2d(ch, ch, tuple(kernel_size), tuple(stride), tuple(padding)))
            layers.extend([nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True)])
        layers.extend([nn.Conv2d(ch, ch*2, (8, 9), 1, 0), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, inplace=True)])
        ch *= 2 
        self.encoder = nn.Sequential(*layers)

        layers = []
        for i in range(10):
            if i % 2 == 0:
                layers.append(nn.Upsample(scale_factor=5 if i == 0 else 2, mode=upsample_mode))
            out_ch = ch if i == 0 or i % 2 == 1 else ch // 2
            layers.extend([nn.Conv2d(ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True)])
            ch = out_ch
        self.decoder = nn.Sequential(*layers)

        self.output = nn.Sequential(nn.Conv2d(ch, 1, 3, 1, 1), nn.BatchNorm2d(1), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        return self.output(x)


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=False):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock3D(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=False):
        super(ConvBlock3D,self).__init__()
        layers = [nn.Conv3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout3d(0.5))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock3DUQ(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=True):
        super(ConvBlock3DUQ,self).__init__()
        layers = [nn.Conv3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(F.dropout3d(p=0.5,training=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', dropout=None):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        if dropout:
            layers.append(nn.Dropout2d(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock3D_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', dropout=False):
        super(ConvBlock3D_Tanh, self).__init__()
        layers = [nn.Conv3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        if dropout:
            layers.append(nn.Dropout3d(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn', dropout=False):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.5))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock3D(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn', dropout=False):
        super(DeconvBlock3D, self).__init__()
        layers = [nn.ConvTranspose3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout3d(0.5))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock3DUQ(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn', dropout=True):
        super(DeconvBlock3DUQ, self).__init__()
        layers = [nn.ConvTranspose3d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(F.dropout3d(p=0.5,training=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class FCN4_Deep(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(FCN4_Deep, self).__init__()
        self.convblock1 = Conv2DwithBN(10, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, stride=2)
        self.convblock4_2 = Conv2DwithBN(dim3, dim3)
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, 7), padding=0)
        
        self.deconv1_1 = Deconv2DwithBN(dim5, dim5, kernel_size=7)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 100)
        x = self.convblock2_1(x) # (None, 64, 250, 100)
        x = self.convblock2_2(x) # (None, 64, 250, 100)
        x = self.convblock3_1(x) # (None, 64, 125, 100)
        x = self.convblock3_2(x) # (None, 64, 125, 100)
        x = self.convblock4_1(x) # (None, 128, 63, 50)
        x = self.convblock4_2(x) # (None, 128, 63, 50)
        x = self.convblock5_1(x) # (None, 128, 32, 25)
        x = self.convblock5_2(x) # (None, 128, 32, 25)
        x = self.convblock6_1(x) # (None, 256, 16, 13)
        x = self.convblock6_2(x) # (None, 256, 16, 13)
        x = self.convblock7_1(x) # (None, 256, 8, 7)
        x = self.convblock7_2(x) # (None, 256, 8, 7)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 7, 7)
        x = self.deconv1_2(x) # (None, 512, 7, 7)
        x = self.deconv2_1(x) # (None, 256, 14, 14)
        x = self.deconv2_2(x) # (None, 256, 14, 14)
        x = self.deconv3_1(x) # (None, 128, 28, 28)
        x = self.deconv3_2(x) # (None, 128, 28, 28)
        x = self.deconv4_1(x) # (None, 64, 56, 56)
        x = self.deconv4_2(x) # (None, 64, 56, 56)
        x = self.deconv5_1(x) # (None, 32, 112, 112)
        x = self.deconv5_2(x) # (None, 32, 112, 112)
        x = F.pad(x, [-6, -6, -6, -6], mode="constant", value=0) # (None, 32, 100, 100)
        x = self.deconv6(x) # (None, 1, 100, 100)
        return x


class FCN4_Deep_Resize(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, upsample_mode='nearest'):
        super(FCN4_Deep_Resize, self).__init__()
        self.convblock1 = Conv2DwithBN(10, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, stride=2)
        self.convblock4_2 = Conv2DwithBN(dim3, dim3)
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, 7), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=7, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 100)
        x = self.convblock2_1(x) # (None, 64, 250, 100)
        x = self.convblock2_2(x) # (None, 64, 250, 100)
        x = self.convblock3_1(x) # (None, 64, 125, 100)
        x = self.convblock3_2(x) # (None, 64, 125, 100)
        x = self.convblock4_1(x) # (None, 128, 63, 50)
        x = self.convblock4_2(x) # (None, 128, 63, 50)
        x = self.convblock5_1(x) # (None, 128, 32, 25)
        x = self.convblock5_2(x) # (None, 128, 32, 25)
        x = self.convblock6_1(x) # (None, 256, 16, 13)
        x = self.convblock6_2(x) # (None, 256, 16, 13)
        x = self.convblock7_1(x) # (None, 256, 8, 7)
        x = self.convblock7_2(x) # (None, 256, 8, 7)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 7, 7)
        x = self.deconv1_2(x) # (None, 512, 7, 7)
        x = self.deconv2_1(x) # (None, 256, 14, 14)
        x = self.deconv2_2(x) # (None, 256, 14, 14)
        x = self.deconv3_1(x) # (None, 128, 28, 28)
        x = self.deconv3_2(x) # (None, 128, 28, 28)
        x = self.deconv4_1(x) # (None, 64, 56, 56)
        x = self.deconv4_2(x) # (None, 64, 56, 56)
        x = self.deconv5_1(x) # (None, 32, 112, 112)
        x = self.deconv5_2(x) # (None, 32, 112, 112)
        x = F.pad(x, [-6, -6, -6, -6], mode="constant", value=0) # (None, 32, 100, 100)
        x = self.deconv6(x) # (None, 1, 100, 100)
        return x


class FCN4_Deep_2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0):
        super(FCN4_Deep_2, self).__init__()
        self.convblock1 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(70 * ratio / 8)), padding=0)
        
        self.deconv1_1 = Deconv2DwithBN(dim5, dim5, kernel_size=5)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class FCN4_V2S_Deep(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, dim6=1024):
        super(FCN4_V2S_Deep, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=7, stride=2, padding=3)
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, stride=2) 
        self.convblock2_2 = Conv2DwithBN(dim2, dim2)
        self.convblock3_1 = Conv2DwithBN(dim2, dim3, stride=2) 
        self.convblock3_2 = Conv2DwithBN(dim3, dim3)
        self.convblock4_1 = Conv2DwithBN(dim3, dim4, stride=2) 
        self.convblock4_2 = Conv2DwithBN(dim4, dim4)
        self.convblock5 = Conv2DwithBN(dim4, dim5, kernel_size=5, padding=0) 
        
        self.deconv1_1 = Deconv2DwithBN(dim5, dim5, kernel_size=(32, 3)) 
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6_1 = Deconv2DwithBN(dim1, dim0, kernel_size=4, stride=2, padding=1)
        self.deconv6_2 = Conv2DwithBN(dim0, dim0)
        self.deconv7 = Conv2DwithBN_Tanh(dim0, 5)

        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 35, 35)
        x = self.convblock2_1(x) # (None, 64, 18, 18)
        x = self.convblock2_2(x) # (None, 64, 18, 18)
        x = self.convblock3_1(x) # (None, 128, 9, 9)
        x = self.convblock3_2(x) # (None, 128, 9, 9)
        x = self.convblock4_1(x) # (None, 256, 5, 5)
        x = self.convblock4_2(x) # (None, 256, 5, 5)
        x = self.convblock5(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 32, 3)
        x = self.deconv1_2(x) # (None, 512, 32, 3)
        x = self.deconv2_1(x) # (None, 256, 64, 6)
        x = self.deconv2_2(x) # (None, 256, 64, 6)
        x = self.deconv3_1(x) # (None, 128, 128, 12)
        x = self.deconv3_2(x) # (None, 128, 128, 12)
        x = self.deconv4_1(x) # (None, 64, 256, 24)
        x = self.deconv4_2(x) # (None, 64, 256, 24)
        x = self.deconv5_1(x) # (None, 32, 512, 48)
        x = self.deconv5_2(x) # (None, 32, 512, 48)
        x = self.deconv6_1(x) # (None, 16, 1024, 96)
        x = self.deconv6_2(x) # (None, 16, 1024, 96)
        x = F.pad(x, [-13, -13, -12, -12], mode="constant", value=0) # (None, 16, 1024, 70)
        x = self.deconv7(x) # (None, 1, 1000, 70)
        return x


class FCN4_V2S_Deep_2(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, dim6=1024):
        super(FCN4_V2S_Deep_2, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=7, stride=2, padding=3)
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, stride=2) 
        self.convblock2_2 = Conv2DwithBN(dim2, dim2)
        self.convblock3_1 = Conv2DwithBN(dim2, dim3, stride=2) 
        self.convblock3_2 = Conv2DwithBN(dim3, dim3)
        self.convblock4_1 = Conv2DwithBN(dim3, dim4, stride=2) 
        self.convblock4_2 = Conv2DwithBN(dim4, dim4)
        self.convblock5_1 = Conv2DwithBN(dim4, dim5, stride=2) 
        self.convblock5_2 = Conv2DwithBN(dim5, dim5)
        self.convblock6 = Conv2DwithBN(dim5, dim6, kernel_size=3, padding=0) 
        
        # self.deconv1_1 = Deconv2DwithBN(dim6, dim6, kernel_size=(16, 3)) 
        # self.deconv1_2 = Conv2DwithBN(dim6, dim6)
        # self.deconv2_1 = Deconv2DwithBN(dim6, dim5, kernel_size=4, stride=2, padding=1)
        # self.deconv2_2 = Conv2DwithBN(dim5, dim5)
        # self.deconv3_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        # self.deconv3_2 = Conv2DwithBN(dim4, dim4)
        # self.deconv4_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        # self.deconv4_2 = Conv2DwithBN(dim3, dim3)
        # self.deconv5_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        # self.deconv5_2 = Conv2DwithBN(dim2, dim2)
        # self.deconv6_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        # self.deconv6_2 = Conv2DwithBN(dim1, dim1)
        # self.deconv7_1 = Deconv2DwithBN(dim1, dim0, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        # self.deconv7_2 = Conv2DwithBN(dim0, dim0)
        # self.deconv8 = Conv2DwithBN_Tanh(dim0, 5)

        self.deconv1_1 = Deconv2DwithBN(dim6, dim5, kernel_size=(32, 3)) 
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6_1 = Deconv2DwithBN(dim1, dim0, kernel_size=4, stride=2, padding=1)
        self.deconv6_2 = Conv2DwithBN(dim0, dim0)
        self.deconv7 = Conv2DwithBN_Tanh(dim0, 5)

        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 35, 35)
        x = self.convblock2_1(x) # (None, 64, 18, 18)
        x = self.convblock2_2(x) # (None, 64, 18, 18)
        x = self.convblock3_1(x) # (None, 128, 9, 9)
        x = self.convblock3_2(x) # (None, 128, 9, 9)
        x = self.convblock4_1(x) # (None, 256, 5, 5)
        x = self.convblock4_2(x) # (None, 256, 5, 5)
        x = self.convblock5_1(x) # (None, 512, 3, 3)
        x = self.convblock5_2(x) # (None, 512, 3, 3)
        x = self.convblock6(x) # (None, 1024, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 1024, 16, 3)
        x = self.deconv1_2(x) # (None, 1024, 16, 3)
        x = self.deconv2_1(x) # (None, 512, 32, 6)
        x = self.deconv2_2(x) # (None, 512, 32, 6)
        x = self.deconv3_1(x) # (None, 256, 64, 12)
        x = self.deconv3_2(x) # (None, 256, 64, 12)
        x = self.deconv4_1(x) # (None, 128, 128, 24)
        x = self.deconv4_2(x) # (None, 128, 128, 24)
        x = self.deconv5_1(x) # (None, 64, 256, 48)
        x = self.deconv5_2(x) # (None, 64, 256, 48)
        x = self.deconv6_1(x) # (None, 32, 512, 96)
        x = self.deconv6_2(x) # (None, 32, 512, 96)
        # x = self.deconv7_1(x) # (None, 16, 1024, 96)
        # x = self.deconv7_2(x) # (None, 16, 1024, 96)
        x = F.pad(x, [-13, -13, -12, -12], mode="constant", value=0) # (None, 16, 1024, 70)
        x = self.deconv7(x) # (None, 1, 1000, 70)
        return x


class ViT_Decoder(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(ViT_Decoder, self).__init__()
        self.vit = ViT(   
            image_size = (1000, 70),
            patch_size = (100, 10),
            num_classes = 512,
            channels = 5,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        self.deconv1_1 = Deconv2DwithBN(dim5, dim5, kernel_size=5)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.vit(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class SeisT_Decoder(nn.Module):
    def __init__(self, depth=6, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(SeisT_Decoder, self).__init__()
        self.seist = SeisT(
            nx = 70, 
            nt = 1000,    
            num_classes = 512,
            channels = 5,
            dim = 1024,
            depth = depth,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        self.deconv1_1 = Deconv2DwithBN(dim5, dim5, kernel_size=5)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.seist(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class ViT_Decoder_Resize(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, upsample_mode='nearest'):
        super(ViT_Decoder_Resize, self).__init__()
        self.vit = ViT(   
            image_size = (1000, 70),
            patch_size = (100, 10),
            num_classes = 512,
            channels = 5,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.vit(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class SeisT_Decoder_Resize(nn.Module):
    def __init__(self, depth=6, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, upsample_mode='nearest'):
        super(SeisT_Decoder_Resize, self).__init__()
        self.seist = SeisT(   
            nx = 70, 
            nt = 1000,    
            num_classes = 512,
            channels = 5,
            dim = 1024,
            depth = depth,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.seist(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class MLP_Mixer_Decoder(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(MLP_Mixer_Decoder, self).__init__()
        self.mlp_mixer = MLPMixer(
            image_size=(1000, 70),
            channels = 5,
            patch_size=(100, 10),
            dim=512,
            depth=12,
            num_classes=512
        )
        
        self.deconv1_1 = Deconv2DwithBN(dim5, dim5, kernel_size=5)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.mlp_mixer(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class MLP_Mixer_Decoder_Resize(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, upsample_mode='nearest'):
        super(MLP_Mixer_Decoder_Resize, self).__init__()
        self.mlp_mixer = MLPMixer(
            image_size=(1000, 70),
            channels = 5,
            patch_size=(100, 10),
            dim=512,
            depth=12,
            num_classes=512
        )
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.mlp_mixer(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class FCN4_Deep_Resize_2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2, self).__init__()
        self.convblock1 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(70 * ratio / 8)), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x



class FCN4_Deep_3(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, dim6=1024):
        super(FCN4_Deep_3, self).__init__()
        self.convblock1 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim5, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim5, dim5)
        self.convblock8 = Conv2DwithBN(dim5, dim6, kernel_size=(8, 9), padding=0)
        
        self.deconv1_1 = Deconv2DwithBN(dim6, dim5, kernel_size=5)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 512, 8, 9)
        x = self.convblock7_2(x) # (None, 512, 8, 9)
        x = self.convblock8(x) # (None, 1024, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class FCN4_Deep_4(nn.Module):
    def __init__(self, dim0=32, dim1=64, dim2=128, dim3=256, dim4=512, dim5=1024, dim6=2048):
        super(FCN4_Deep_4, self).__init__()
        self.convblock1 = Conv2DwithBN(10, dim0, kernel_size=7, stride=1, padding=3)
        self.convblock2_1 = Conv2DwithBN(dim0, dim1, stride=2)
        self.convblock2_2 = Conv2DwithBN(dim1, dim1)
        self.convblock3_1 = Conv2DwithBN(dim1, dim2, stride=2)
        self.convblock3_2 = Conv2DwithBN(dim2, dim2)
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, stride=2)
        self.convblock4_2 = Conv2DwithBN(dim3, dim3)
        self.convblock5_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim4, dim4)
        self.convblock6_1 = Conv2DwithBN(dim4, dim5, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim5, dim5)
        self.convblock7 = Conv2DwithBN(dim5, dim6, kernel_size=7, padding=0)
        
        self.deconv1_1 = Deconv2DwithBN(dim6, dim5, kernel_size=7)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6_1 = Deconv2DwithBN(dim1, dim0, kernel_size=4, stride=2, padding=1)
        self.deconv6_2 = Conv2DwithBN(dim0, dim0)
        self.deconv7 = Conv2DwithBN_Tanh(dim0, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 200, 200)
        x = self.convblock2_1(x) # (None, 64, 100, 100)
        x = self.convblock2_2(x) # (None, 64, 100, 100)
        x = self.convblock3_1(x) # (None, 128, 50, 50)
        x = self.convblock3_2(x) # (None, 128, 50, 50)
        x = self.convblock4_1(x) # (None, 256, 25, 25)
        x = self.convblock4_2(x) # (None, 256, 25, 25)
        x = self.convblock5_1(x) # (None, 512, 13, 13)
        x = self.convblock5_2(x) # (None, 512, 13, 13)
        x = self.convblock6_1(x) # (None, 1024, 7, 7)
        x = self.convblock6_2(x) # (None, 1024, 7, 7)
        x = self.convblock7(x) # (None, 2048, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 1024, 7, 7)
        x = self.deconv1_2(x) # (None, 1024, 7, 7)
        x = self.deconv2_1(x) # (None, 512, 14, 14)
        x = self.deconv2_2(x) # (None, 512, 14, 14)
        x = self.deconv3_1(x) # (None, 256, 28, 28)
        x = self.deconv3_2(x) # (None, 256, 28, 28)
        x = self.deconv4_1(x) # (None, 128, 56, 56)
        x = self.deconv4_2(x) # (None, 128, 56, 56)
        x = self.deconv5_1(x) # (None, 64, 112, 112)
        x = self.deconv5_2(x) # (None, 64, 112, 112)
        x = self.deconv6_1(x) # (None, 32, 224, 224)
        x = self.deconv6_2(x) # (None, 32, 224, 224)
        x = F.pad(x, [-12, -12, -12, -12], mode="constant", value=0) # (None, 16, 200, 200)
        x = self.deconv7(x) # (None, 1, 200, 200)
        return x

# 600, 60 --> 40, 60
class FCN4_Salt(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(FCN4_Salt, self).__init__()
        self.convblock11 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock12 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock21 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock22 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock31 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock32 = Conv2DwithBN(dim2, dim3, stride=2)
        self.convblock41 = Conv2DwithBN(dim3, dim3)
        self.convblock42 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock51 = Conv2DwithBN(dim3, dim3)
        self.convblock52 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock61 = Conv2DwithBN(dim4, dim4)
        self.convblock62 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7 = Conv2DwithBN(dim4, dim5, kernel_size=(5, 4), padding=0)
        self.deconv11 = Deconv2DwithBN(dim5, dim5, kernel_size=(3, 4))
        self.deconv12 = Conv2DwithBN(dim5, dim5)
        self.deconv21 = Deconv2DwithBN(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv22 = Conv2DwithBN(dim4, dim4)
        self.deconv31 = Deconv2DwithBN(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv32 = Conv2DwithBN(dim3, dim3)
        self.deconv41 = Deconv2DwithBN(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv42 = Conv2DwithBN(dim2, dim2)
        self.deconv51 = Deconv2DwithBN(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv52 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)

    def forward(self,x):
        x = self.convblock11(x) # (None, 32, 300, 60)
        x = self.convblock12(x) # (None, 64, 150, 60)
        x = self.convblock21(x) # (None, 64, 150, 60)
        x = self.convblock22(x) # (None, 64, 75, 60)
        x = self.convblock31(x) # (None, 64, 75, 60)
        x = self.convblock32(x) # (None, 128, 38, 30)
        x = self.convblock41(x) # (None, 128, 38, 30)
        x = self.convblock42(x) # (None, 128, 19, 15)
        x = self.convblock51(x) # (None, 128, 19, 15)
        x = self.convblock52(x) # (None, 256, 10, 8)
        x = self.convblock61(x) # (None, 256, 10, 8)
        x = self.convblock62(x) # (None, 256, 5, 4)
        x = self.convblock7(x) # (None, 512, 1, 1)
        x = self.deconv11(x) # (None, 512, 3, 4)
        x = self.deconv12(x) # (None, 512, 3, 4)
        x = self.deconv21(x) # (None, 256, 6, 8)
        x = self.deconv22(x) # (None, 256, 6, 8)
        x = self.deconv31(x) # (None, 128, 12, 16)
        x = self.deconv32(x) # (None, 128, 12, 16)
        x = self.deconv41(x) # (None, 64, 24, 32)
        x = self.deconv42(x) # (None, 64, 24, 32)
        x = self.deconv51(x) # (None, 32, 48, 64)
        x = self.deconv52(x) # (None, 32, 48, 64)
        x = F.pad(x, pad=[-2, -2, -4, -4], mode="constant", value=0)
        x = self.deconv6(x)
        return x


class FCN4_Salt_Resize(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, upsample_mode='nearest'):
        super(FCN4_Salt_Resize, self).__init__()
        self.convblock11 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock12 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock21 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock22 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock31 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock32 = Conv2DwithBN(dim2, dim3, stride=2)
        self.convblock41 = Conv2DwithBN(dim3, dim3)
        self.convblock42 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock51 = Conv2DwithBN(dim3, dim3)
        self.convblock52 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock61 = Conv2DwithBN(dim4, dim4)
        self.convblock62 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7 = Conv2DwithBN(dim4, dim5, kernel_size=(5, 4), padding=0)
        self.deconv11 = ResizeConv2DwithBN(dim5, dim5, scale_factor=(3, 4), mode=upsample_mode)
        self.deconv12 = Conv2DwithBN(dim5, dim5)
        self.deconv21 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv22 = Conv2DwithBN(dim4, dim4)
        self.deconv31 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv32 = Conv2DwithBN(dim3, dim3)
        self.deconv41 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv42 = Conv2DwithBN(dim2, dim2)
        self.deconv51 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv52 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
    def forward(self,x):
        x = self.convblock11(x)
        x = self.convblock12(x)
        x = self.convblock21(x)
        x = self.convblock22(x)
        x = self.convblock31(x)
        x = self.convblock32(x)
        x = self.convblock41(x)
        x = self.convblock42(x)
        x = self.convblock51(x)
        x = self.convblock52(x)
        x = self.convblock61(x)
        x = self.convblock62(x)
        x = self.convblock7(x)
        x = self.deconv11(x)
        x = self.deconv12(x)
        x = self.deconv21(x)
        x = self.deconv22(x)
        x = self.deconv31(x)
        x = self.deconv32(x)
        x = self.deconv41(x)
        x = self.deconv42(x)
        x = self.deconv51(x)
        x = self.deconv52(x)
        x = F.pad(x, pad=[-2, -2, -4, -4], mode="constant", value=0)
        x = self.deconv6(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256):
        super(Discriminator, self).__init__()
        self.convblock1_1 = Conv2DwithBN(1, dim1, stride=2)
        self.convblock1_2 = Conv2DwithBN(dim1, dim1)
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, stride=2)
        self.convblock2_2 = Conv2DwithBN(dim2, dim2)
        self.convblock3_1 = Conv2DwithBN(dim2, dim3, stride=2)
        self.convblock3_2 = Conv2DwithBN(dim3, dim3)
        self.convblock4_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock4_2 = Conv2DwithBN(dim4, dim4)
        self.convblock5 = Conv2DwithBN(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x


class Conv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=None, stride=None, padding=None):
        super(Conv_HPGNN, self).__init__()
        layers = [
            Conv2DwithBN(in_fea, out_fea, relu_slop=0.1, dropout=False),
            Conv2DwithBN(out_fea, out_fea, relu_slop=0.1, dropout=False),
        ]
        if kernel_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Deconv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size):
        super(Deconv_HPGNN, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_fea, in_fea, kernel_size=kernel_size, stride=2, padding=0),
            Conv2DwithBN(in_fea, out_fea, relu_slop=0.1, dropout=False),
            Conv2DwithBN(out_fea, out_fea, relu_slop=0.1, dropout=False)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)





class HPGNN(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512):
        super(HPGNN, self).__init__()
        self.convblock1 = nn.Conv2d(5, dim0, kernel_size=1, stride=1, padding=0)
        self.convblock2 = Conv_HPGNN(dim0, dim1, kernel_size=(4, 2), stride=(4, 2), padding=(2, 0))
        self.convblock3 = Conv_HPGNN(dim1, dim2, kernel_size=(4, 2), stride=(4, 2), padding=(2, 0))
        self.convblock4 = Conv_HPGNN(dim2, dim3, kernel_size=(4, 2), stride=(4, 2), padding=(2, 0))
        self.convblock5 = Conv_HPGNN(dim3, dim4, kernel_size=(4, 2), stride=(4, 2), padding=0)
        self.convblock6 = Conv_HPGNN(dim4, dim5)
        self.deconv1 = Deconv_HPGNN(dim5, dim4, 2)
        self.deconv2 = Deconv_HPGNN(dim4, dim3, 3)
        self.deconv3 = Deconv_HPGNN(dim3, dim2, 3)
        self.deconv4 = Deconv_HPGNN(dim2, dim1, 2)
        self.conv_last = nn.Conv2d(dim1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.conv_last(x)
        return torch.clamp(F.relu(x), 0.0, 1.0)


class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 32, 28
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 64, 56
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80) 128, 112
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

class PDE_Regress(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(PDE_Regress, self).__init__()
        self.convblock1 = ConvBlock(2, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=2, padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=2, padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=2, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        
        self.gap = nn.AdaptiveAvgPool2d((512,1))
        self.linear = nn.Linear(512,3)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 70, 70)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 35, 35)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 35, 35)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 18, 18)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 18, 18)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 9, 9) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 9, 9)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 5, 5) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 5, 5)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 3, 3) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 3, 3)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 2, 2) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 2, 2)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())
        
        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
        x = self.linear(x)

        return x

class PDE_Regress_CO2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(PDE_Regress_CO2, self).__init__()
        self.convblock1 = ConvBlock(4, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=(2,4), padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=(2,3), padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=3, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        
        self.gap = nn.AdaptiveAvgPool2d((512,1))
        self.linear = nn.Linear(512,5)
        




    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 141, 401)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 71, 101)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 71, 101)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 36, 34)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 36, 34)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 18, 17) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 18, 17)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 9, 9) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 9, 9)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 5, 5) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 5, 5)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 3, 3) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 3, 3)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())
        
        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
        x = self.linear(x)

        return x

class PDE_Regress_CO2_UNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, upsample_mode='nearest', **kwargs):
        super(PDE_Regress_CO2_UNet, self).__init__()
        self.convblock1 = ConvBlock(2, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=(2,4), padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=(2,3), padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=3, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=(5,10), mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=(4,4), mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=(2,3), mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 3)
#        self.gap = nn.AdaptiveAvgPool2d((512,1))
#        self.linear = nn.Linear(512,5)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 141, 401)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 71, 101)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 71, 101)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 36, 34)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 36, 34)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 18, 17) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 18, 17)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 9, 9) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 9, 9)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 5, 5) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 5, 5)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 3, 3) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 3, 3)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())
        
        x = self.deconv1_1(x) # (None, 512, 5, 10)
#        print('d1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 5, 10)
#        print('d1_2:',x.size())
        x = self.deconv2_1(x) # (None, 256, 20, 40)
#        print('d2_1:',x.size())
        x = self.deconv2_2(x) # (None, 256, 20, 40)
#        print('d2_2:',x.size())
        x = self.deconv3_1(x) # (None, 128, 40, 120)
#        print('d3_1:',x.size())
        x = self.deconv3_2(x) # (None, 128, 40, 120)
#        print('d3_2:',x.size())
        x = self.deconv4_1(x) # (None, 64, 80, 240)
#        print('d4_1:',x.size())
        x = self.deconv4_2(x) # (None, 64, 80, 240)
#        print('d4_2:',x.size())
        x = self.deconv5_1(x) # (None, 32, 160, 480)
#        print('d5_1:',x.size())
        x = self.deconv5_2(x) # (None, 32, 160, 480)
#        print('d5_2:',x.size())
        x = F.pad(x, [-40, -39, -10, -9], mode="constant", value=0) # (None, 32, 141, 401)
#        print('cut:',x.size())
        x = self.deconv6(x) # (None, 1, 70, 70)
#        print('d6:',x.size())
#        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
#        x = self.linear(x)

        return x
class PDE_Regress_CO2_UNet_dv_all_labels_diff(nn.Module):
    def __init__(self, dim1=64, dim2=96, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, upsample_mode='nearest', **kwargs):
        super(PDE_Regress_CO2_UNet_dv_all_labels_diff, self).__init__()
        self.convblock1 = ConvBlock(45, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=(2,4), padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=(2,3), padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=3, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=(5,10), mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=(4,4), mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=(2,3), mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 5)
#        self.gap = nn.AdaptiveAvgPool2d((512,1))
#        self.linear = nn.Linear(512,5)

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 141, 401)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 71, 101)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 71, 101)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 36, 34)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 36, 34)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 18, 17) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 18, 17)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 9, 9) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 9, 9)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 5, 5) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 5, 5)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 3, 3) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 3, 3)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())

        x = self.deconv1_1(x) # (None, 512, 5, 10)
#        print('d1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 5, 10)
#        print('d1_2:',x.size())
        x = self.deconv2_1(x) # (None, 256, 20, 40)
#        print('d2_1:',x.size())
        x = self.deconv2_2(x) # (None, 256, 20, 40)
#        print('d2_2:',x.size())
        x = self.deconv3_1(x) # (None, 128, 40, 120)
#        print('d3_1:',x.size())
        x = self.deconv3_2(x) # (None, 128, 40, 120)
#        print('d3_2:',x.size())
        x = self.deconv4_1(x) # (None, 64, 80, 240)
#        print('d4_1:',x.size())
        x = self.deconv4_2(x) # (None, 64, 80, 240)
#        print('d4_2:',x.size())
        x = self.deconv5_1(x) # (None, 32, 160, 480)
#        print('d5_1:',x.size())
        x = self.deconv5_2(x) # (None, 32, 160, 480)
#        print('d5_2:',x.size())
        x = F.pad(x, [-40, -39, -10, -9], mode="constant", value=0) # (None, 32, 141, 401)
#        print('cut:',x.size())
        x = self.deconv6(x) # (None, 1, 70, 70)
#        print('d6:',x.size())
#        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
#        x = self.linear(x)

        return x

class PDE_Regress_CO2_UNet_dv_all_labels(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, upsample_mode='nearest', **kwargs):
        super(PDE_Regress_CO2_UNet_dv_all_labels, self).__init__()
        self.convblock1 = ConvBlock(9, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=(2,4), padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=(2,3), padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=3, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=(5,10), mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=(4,4), mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=(2,3), mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 5)
#        self.gap = nn.AdaptiveAvgPool2d((512,1))
#        self.linear = nn.Linear(512,5)

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 141, 401)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 71, 101)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 71, 101)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 36, 34)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 36, 34)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 18, 17) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 18, 17)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 9, 9) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 9, 9)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 5, 5) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 5, 5)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 3, 3) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 3, 3)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())

        x = self.deconv1_1(x) # (None, 512, 5, 10)
#        print('d1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 5, 10)
#        print('d1_2:',x.size())
        x = self.deconv2_1(x) # (None, 256, 20, 40)
#        print('d2_1:',x.size())
        x = self.deconv2_2(x) # (None, 256, 20, 40)
#        print('d2_2:',x.size())
        x = self.deconv3_1(x) # (None, 128, 40, 120)
#        print('d3_1:',x.size())
        x = self.deconv3_2(x) # (None, 128, 40, 120)
#        print('d3_2:',x.size())
        x = self.deconv4_1(x) # (None, 64, 80, 240)
#        print('d4_1:',x.size())
        x = self.deconv4_2(x) # (None, 64, 80, 240)
#        print('d4_2:',x.size())
        x = self.deconv5_1(x) # (None, 32, 160, 480)
#        print('d5_1:',x.size())
        x = self.deconv5_2(x) # (None, 32, 160, 480)
#        print('d5_2:',x.size())
        x = F.pad(x, [-40, -39, -10, -9], mode="constant", value=0) # (None, 32, 141, 401)
#        print('cut:',x.size())
        x = self.deconv6(x) # (None, 1, 70, 70)
#        print('d6:',x.size())
#        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
#        x = self.linear(x)

        return x

class PDE_Regress_CO2_UNet_dv_all_labels_o3(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, upsample_mode='nearest', **kwargs):
        super(PDE_Regress_CO2_UNet_dv_all_labels_o3, self).__init__()
        self.convblock1 = ConvBlock(9, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=(2,4), padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=(2,3), padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=3, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=(5,10), mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=(4,4), mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=(2,3), mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 3)
#        self.gap = nn.AdaptiveAvgPool2d((512,1))
#        self.linear = nn.Linear(512,5)

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 141, 401)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 71, 101)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 71, 101)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 36, 34)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 36, 34)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 18, 17) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 18, 17)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 9, 9) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 9, 9)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 5, 5) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 5, 5)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 3, 3) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 3, 3)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())

        x = self.deconv1_1(x) # (None, 512, 5, 10)
#        print('d1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 5, 10)
#        print('d1_2:',x.size())
        x = self.deconv2_1(x) # (None, 256, 20, 40)
#        print('d2_1:',x.size())
        x = self.deconv2_2(x) # (None, 256, 20, 40)
#        print('d2_2:',x.size())
        x = self.deconv3_1(x) # (None, 128, 40, 120)
#        print('d3_1:',x.size())
        x = self.deconv3_2(x) # (None, 128, 40, 120)
#        print('d3_2:',x.size())
        x = self.deconv4_1(x) # (None, 64, 80, 240)
#        print('d4_1:',x.size())
        x = self.deconv4_2(x) # (None, 64, 80, 240)
#        print('d4_2:',x.size())
        x = self.deconv5_1(x) # (None, 32, 160, 480)
#        print('d5_1:',x.size())
        x = self.deconv5_2(x) # (None, 32, 160, 480)
#        print('d5_2:',x.size())
        x = F.pad(x, [-40, -39, -10, -9], mode="constant", value=0) # (None, 32, 141, 401)
#        print('cut:',x.size())
        x = self.deconv6(x) # (None, 1, 70, 70)
#        print('d6:',x.size())
#        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
#        x = self.linear(x)

        return x

class PDE_Regress_CO2_UNet_dv_all_labels_diff_o3(nn.Module):
    def __init__(self, dim1=64, dim2=96, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, upsample_mode='nearest', **kwargs):
        super(PDE_Regress_CO2_UNet_dv_all_labels_diff_o3, self).__init__()
        self.convblock1 = ConvBlock(45, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=(2,4), padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=(2,3), padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=3, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=(5,10), mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=(4,4), mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=(2,3), mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 3)
#        self.gap = nn.AdaptiveAvgPool2d((512,1))
#        self.linear = nn.Linear(512,5)

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 141, 401)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 71, 101)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 71, 101)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 36, 34)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 36, 34)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 18, 17) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 18, 17)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 9, 9) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 9, 9)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 5, 5) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 5, 5)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 3, 3) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 3, 3)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())

        x = self.deconv1_1(x) # (None, 512, 5, 10)
#        print('d1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 5, 10)
#        print('d1_2:',x.size())
        x = self.deconv2_1(x) # (None, 256, 20, 40)
#        print('d2_1:',x.size())
        x = self.deconv2_2(x) # (None, 256, 20, 40)
#        print('d2_2:',x.size())
        x = self.deconv3_1(x) # (None, 128, 40, 120)
#        print('d3_1:',x.size())
        x = self.deconv3_2(x) # (None, 128, 40, 120)
#        print('d3_2:',x.size())
        x = self.deconv4_1(x) # (None, 64, 80, 240)
#        print('d4_1:',x.size())
        x = self.deconv4_2(x) # (None, 64, 80, 240)
#        print('d4_2:',x.size())
        x = self.deconv5_1(x) # (None, 32, 160, 480)
#        print('d5_1:',x.size())
        x = self.deconv5_2(x) # (None, 32, 160, 480)
#        print('d5_2:',x.size())
        x = F.pad(x, [-40, -39, -10, -9], mode="constant", value=0) # (None, 32, 141, 401)
#        print('cut:',x.size())
        x = self.deconv6(x) # (None, 1, 70, 70)
#        print('d6:',x.size())
#        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
#        x = self.linear(x)

        return x

class PDE_Regress_CO2_UNet_dv_all_labels_i10o3(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, upsample_mode='nearest', **kwargs):
        super(PDE_Regress_CO2_UNet_dv_all_labels_i10o3, self).__init__()
        self.convblock1 = ConvBlock(10, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=(2,4), padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=(2,3), padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=3, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=(5,10), mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=(4,4), mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=(2,3), mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 3)
#        self.gap = nn.AdaptiveAvgPool2d((512,1))
#        self.linear = nn.Linear(512,5)

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 141, 401)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 71, 101)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 71, 101)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 36, 34)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 36, 34)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 18, 17) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 18, 17)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 9, 9) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 9, 9)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 5, 5) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 5, 5)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 3, 3) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 3, 3)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())

        x = self.deconv1_1(x) # (None, 512, 5, 10)
#        print('d1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 5, 10)
#        print('d1_2:',x.size())
        x = self.deconv2_1(x) # (None, 256, 20, 40)
#        print('d2_1:',x.size())
        x = self.deconv2_2(x) # (None, 256, 20, 40)
#        print('d2_2:',x.size())
        x = self.deconv3_1(x) # (None, 128, 40, 120)
#        print('d3_1:',x.size())
        x = self.deconv3_2(x) # (None, 128, 40, 120)
#        print('d3_2:',x.size())
        x = self.deconv4_1(x) # (None, 64, 80, 240)
#        print('d4_1:',x.size())
        x = self.deconv4_2(x) # (None, 64, 80, 240)
#        print('d4_2:',x.size())
        x = self.deconv5_1(x) # (None, 32, 160, 480)
#        print('d5_1:',x.size())
        x = self.deconv5_2(x) # (None, 32, 160, 480)
#        print('d5_2:',x.size())
        x = F.pad(x, [-40, -39, -10, -9], mode="constant", value=0) # (None, 32, 141, 401)
#        print('cut:',x.size())
        x = self.deconv6(x) # (None, 1, 70, 70)
#        print('d6:',x.size())
#        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
#        x = self.linear(x)

        return x

class PDE_Regress_CO2_UNet_dv_all_labels_diff_i30o3(nn.Module):
    def __init__(self, dim1=32, dim2=96, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, upsample_mode='nearest', **kwargs):
        super(PDE_Regress_CO2_UNet_dv_all_labels_diff_i30o3, self).__init__()
        self.convblock1 = ConvBlock(30, dim1, kernel_size=3, padding=1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=(2,4), padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=3, stride=(2,3), padding=1)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=3, padding=1)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock5_1 = ConvBlock(dim3, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim3, dim3, kernel_size=3, padding=1)
        self.convblock6_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock6_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock7_1 = ConvBlock(dim4, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock7_2 = ConvBlock(dim4, dim4, kernel_size=3, padding=1)
        self.convblock8 = nn.Conv2d(dim4, dim5, kernel_size=3, stride=3, padding=1)
#ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=(5,10), mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=(4,4), mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=(2,3), mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 3)
#        self.gap = nn.AdaptiveAvgPool2d((512,1))
#        self.linear = nn.Linear(512,5)

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 141, 401)
#        print('1:',x.size())
        x = self.convblock2_1(x) # (None, 64, 71, 101)
#        print('2_1:',x.size())
        x = self.convblock2_2(x) # (None, 64, 71, 101)
#        print('2_2:',x.size())
        x = self.convblock3_1(x) # (None, 64, 36, 34)
#        print('3_1:',x.size())
        x = self.convblock3_2(x) # (None, 64, 36, 34)
#        print('3_2:',x.size())
        x = self.convblock4_1(x) # (None, 128, 18, 17) 
#        print('4_1:',x.size())
        x = self.convblock4_2(x) # (None, 128, 18, 17)
#        print('4_2:',x.size())
        x = self.convblock5_1(x) # (None, 128, 9, 9) 
#        print('5_1:',x.size())
        x = self.convblock5_2(x) # (None, 128, 9, 9)
#        print('5_2:',x.size())
        x = self.convblock6_1(x) # (None, 256, 5, 5) 
#        print('6_1:',x.size())
        x = self.convblock6_2(x) # (None, 256, 5, 5)
#        print('6_2:',x.size())
        x = self.convblock7_1(x) # (None, 256, 3, 3) 
#        print('7_1:',x.size())
        x = self.convblock7_2(x) # (None, 256, 3, 3)
#        print('7_2:',x.size())
        x = self.convblock8(x) # (None, 512, 1, 1)
#        print('8:',x.size())

        x = self.deconv1_1(x) # (None, 512, 5, 10)
#        print('d1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 5, 10)
#        print('d1_2:',x.size())
        x = self.deconv2_1(x) # (None, 256, 20, 40)
#        print('d2_1:',x.size())
        x = self.deconv2_2(x) # (None, 256, 20, 40)
#        print('d2_2:',x.size())
        x = self.deconv3_1(x) # (None, 128, 40, 120)
#        print('d3_1:',x.size())
        x = self.deconv3_2(x) # (None, 128, 40, 120)
#        print('d3_2:',x.size())
        x = self.deconv4_1(x) # (None, 64, 80, 240)
#        print('d4_1:',x.size())
        x = self.deconv4_2(x) # (None, 64, 80, 240)
#        print('d4_2:',x.size())
        x = self.deconv5_1(x) # (None, 32, 160, 480)
#        print('d5_1:',x.size())
        x = self.deconv5_2(x) # (None, 32, 160, 480)
#        print('d5_2:',x.size())
        x = F.pad(x, [-40, -39, -10, -9], mode="constant", value=0) # (None, 32, 141, 401)
#        print('cut:',x.size())
        x = self.deconv6(x) # (None, 1, 70, 70)
#        print('d6:',x.size())
#        x = x.squeeze(dim=3).squeeze(dim=2)
#        x = self.gap(x)
#        x = self.linear(x)

        return x


class FCN4_Deep_Resize_2_mlreal(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2_mlreal, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(31 * ratio / 8)), padding=0)

        self.linblock_1_x = LinearwithBN(512,20)
        self.linblock_1_y = LinearwithBN(512,20)
        self.linblock_1_z = LinearwithBN(512,20)

        self.linblock_2_x = LinearwithBN(20,1)
        self.linblock_2_y = LinearwithBN(20,1)
        self.linblock_2_z = LinearwithBN(20,1)

        #self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 31)
        x = self.convblock2_1(x) # (None, 64, 250, 31)
        x = self.convblock2_2(x) # (None, 64, 250, 31)
        x = self.convblock3_1(x) # (None, 64, 125, 31)
        x = self.convblock3_2(x) # (None, 64, 125, 31)
        x = self.convblock4_1(x) # (None, 128, 63, 31)
        x = self.convblock4_2(x) # (None, 128, 63, 31)
        x = self.convblock5_1(x) # (None, 128, 32, 16)
        x = self.convblock5_2(x) # (None, 128, 32, 16)
        x = self.convblock6_1(x) # (None, 256, 16, 8)
        x = self.convblock6_2(x) # (None, 256, 16, 8)
        x = self.convblock7_1(x) # (None, 256, 8, 4)
        x = self.convblock7_2(x) # (None, 256, 8, 4)
        x = self.convblock8(x) # (None, 512, 1, 1)

        # Decoder Part 
        x = x.squeeze(dim=3).squeeze(dim=2)
        o_1 = self.linblock_1_x(x)
        o_2 = self.linblock_1_y(x)
        o_3 = self.linblock_1_z(x)
        
        o_1 = self.linblock_2_x(o_1)
        o_2 = self.linblock_2_y(o_2)
        o_3 = self.linblock_2_z(o_3)

        return torch.cat((o_1,o_2,o_3),1)

class FCN4_Deep_Resize_2_mlreal_large(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2_mlreal_large, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(31 * ratio / 8)), padding=0)

        self.linblock_1_x = LinearwithBN(512,128)
        self.linblock_1_y = LinearwithBN(512,128)
        self.linblock_1_z = LinearwithBN(512,128)

        self.linblock_2_x = LinearwithBN(128,64)
        self.linblock_2_y = LinearwithBN(128,64)
        self.linblock_2_z = LinearwithBN(128,64)

        self.linblock_3_x = LinearwithBN(64,16)
        self.linblock_3_y = LinearwithBN(64,16)
        self.linblock_3_z = LinearwithBN(64,16)

        self.linblock_4_x = LinearwithBN(16,1)
        self.linblock_4_y = LinearwithBN(16,1)
        self.linblock_4_z = LinearwithBN(16,1)

        #self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 31)
        x = self.convblock2_1(x) # (None, 64, 250, 31)
        x = self.convblock2_2(x) # (None, 64, 250, 31)
        x = self.convblock3_1(x) # (None, 64, 125, 31)
        x = self.convblock3_2(x) # (None, 64, 125, 31)
        x = self.convblock4_1(x) # (None, 128, 63, 31)
        x = self.convblock4_2(x) # (None, 128, 63, 31)
        x = self.convblock5_1(x) # (None, 128, 32, 16)
        x = self.convblock5_2(x) # (None, 128, 32, 16)
        x = self.convblock6_1(x) # (None, 256, 16, 8)
        x = self.convblock6_2(x) # (None, 256, 16, 8)
        x = self.convblock7_1(x) # (None, 256, 8, 4)
        x = self.convblock7_2(x) # (None, 256, 8, 4)
        x = self.convblock8(x) # (None, 512, 1, 1)

        # Decoder Part 
        x = x.squeeze(dim=3).squeeze(dim=2)
        o_1 = self.linblock_1_x(x)
        o_2 = self.linblock_1_y(x)
        o_3 = self.linblock_1_z(x)
        
        o_1 = self.linblock_2_x(o_1)
        o_2 = self.linblock_2_y(o_2)
        o_3 = self.linblock_2_z(o_3)

        o_1 = self.linblock_3_x(o_1)
        o_2 = self.linblock_3_y(o_2)
        o_3 = self.linblock_3_z(o_3)

        o_1 = self.linblock_4_x(o_1)
        o_2 = self.linblock_4_y(o_2)
        o_3 = self.linblock_4_z(o_3)

        return torch.cat((o_1,o_2,o_3),1)



class FCN4_Deep_Resize_2_mlreal_o3(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2_mlreal_o3, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(31 * ratio / 8)), padding=0)

        self.linblock_1 = LinearwithBN(512,20)
        self.linblock_2 = LinearwithBN(20,3)

        #self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 31)
        x = self.convblock2_1(x) # (None, 64, 250, 31)
        x = self.convblock2_2(x) # (None, 64, 250, 31)
        x = self.convblock3_1(x) # (None, 64, 125, 31)
        x = self.convblock3_2(x) # (None, 64, 125, 31)
        x = self.convblock4_1(x) # (None, 128, 63, 31)
        x = self.convblock4_2(x) # (None, 128, 63, 31)
        x = self.convblock5_1(x) # (None, 128, 32, 16)
        x = self.convblock5_2(x) # (None, 128, 32, 16)
        x = self.convblock6_1(x) # (None, 256, 16, 8)
        x = self.convblock6_2(x) # (None, 256, 16, 8)
        x = self.convblock7_1(x) # (None, 256, 8, 4)
        x = self.convblock7_2(x) # (None, 256, 8, 4)
        x = self.convblock8(x) # (None, 512, 1, 1)

        # Decoder Part 
        x = x.squeeze(dim=3).squeeze(dim=2)
        o_1 = self.linblock_1(x)
        o_2 = self.linblock_2(o_1)


        return o_2

class Passive_heat_map(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(Passive_heat_map, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(31 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6_1 = DeconvBlock(dim1, dim0, kernel_size=4, stride=(4,5), padding=1)
        self.deconv6_2 = ConvBlock(dim0, dim0)
        self.deconv7 = ConvBlock_Tanh(dim0, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 32, 28
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 64, 56
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80) 128, 112
        x = self.deconv5_2(x) # (None, 32, 80, 80) 
        #print('size after decon5_2',x.size())
        x = self.deconv6_1(x) # (None, 16, 320, 400) #nx ny 317 395
        x = self.deconv6_2(x) # (None, 16, 320, 400)
        #print('size after decon6_2',x.size()) # 318 397
        x = F.pad(x, [-1, -1, -1, 0], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv7(x) # (None, 1, 70, 70)
        return x.squeeze(dim=1)

class Passive_heat_map_3D(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(Passive_heat_map_3D, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(31 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock3D(dim5, dim5, kernel_size=(4,5,1))
        self.deconv1_2 = ConvBlock3D(dim5, dim5)
        self.deconv2_1 = DeconvBlock3D(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock3D(dim4, dim4)
        self.deconv3_1 = DeconvBlock3D(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock3D(dim3, dim3)
        self.deconv4_1 = DeconvBlock3D(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock3D(dim2, dim2)
        self.deconv5_1 = DeconvBlock3D(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock3D(dim1, dim1)
        #self.deconv6_1 = DeconvBlock3D(dim1, dim0, kernel_size=4, stride=(4,5), padding=1)
        #self.deconv6_2 = ConvBlock(dim0, dim0)
        self.deconv7 = ConvBlock3D_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        x = torch.unsqueeze(x,-1)
        #print('x after conv 8:',x.size())
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 4, 5, 1)
        #print('x after deconv 1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 4, 5, 1)
        x = self.deconv2_1(x) # (None, 256, 8, 10, 2) 
        x = self.deconv2_2(x) # (None, 256, 8, 10, 2)
        x = self.deconv3_1(x) # (None, 128, 16, 20, 4) 
        x = self.deconv3_2(x) # (None, 128, 16, 20, 4)
        x = self.deconv4_1(x) # (None, 64, 32, 40, 8) 
        x = self.deconv4_2(x) # (None, 64, 32, 40, 8)
        x = self.deconv5_1(x) # (None, 32, 64, 80, 16) 
        x = self.deconv5_2(x) # (None, 32, 64, 80, 16) 
        #print('size after decon5_2',x.size())
        #x = self.deconv6_1(x) # (None, 16, 320, 400) #
        #x = self.deconv6_2(x) # (None, 16, 320, 400)
        #print('size after decon6_2',x.size()) # nx ny nz 63 79 11
        x = F.pad(x, [-2, -3, -1, 0, -1, 0], mode="constant", value=0) 
        x = self.deconv7(x) 
        return x.squeeze(dim=1)


class Passive_heat_map_3D_intz(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(Passive_heat_map_3D_intz, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(31 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock3D(dim5, dim5, kernel_size=(4,5,4))
        self.deconv1_2 = ConvBlock3D(dim5, dim5)
        self.deconv2_1 = DeconvBlock3D(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock3D(dim4, dim4)
        self.deconv3_1 = DeconvBlock3D(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock3D(dim3, dim3)
        self.deconv4_1 = DeconvBlock3D(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock3D(dim2, dim2)
        self.deconv5_1 = DeconvBlock3D(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock3D(dim1, dim1)
        #self.deconv6_1 = DeconvBlock3D(dim1, dim0, kernel_size=4, stride=(4,5), padding=1)
        #self.deconv6_2 = ConvBlock(dim0, dim0)
        self.deconv7 = ConvBlock3D_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        x = torch.unsqueeze(x,-1)
        #x = torch.unsqueeze(x,-1)
        #print('x after conv 8:',x.size())
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 4, 5, 4)
        #print('x after deconv 1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 4, 5, 4)
        x = self.deconv2_1(x) # (None, 256, 8, 10, 8) 
        x = self.deconv2_2(x) # (None, 256, 8, 10, 8)
        x = self.deconv3_1(x) # (None, 128, 16, 20, 16) 
        x = self.deconv3_2(x) # (None, 128, 16, 20, 16)
        x = self.deconv4_1(x) # (None, 64, 32, 40, 32) 
        x = self.deconv4_2(x) # (None, 64, 32, 40, 32)
        x = self.deconv5_1(x) # (None, 32, 64, 80, 64) 
        x = self.deconv5_2(x) # (None, 32, 64, 80, 64) 
        #print('size after decon5_2',x.size())
        #x = self.deconv6_1(x) # (None, 16, 320, 400) #
        #x = self.deconv6_2(x) # (None, 16, 320, 400)
        #print('size after decon6_2',x.size()) # nx ny nz 63 79 55
        x = F.pad(x, [-4, -5, -1, 0, -1, 0], mode="constant", value=0) 
        x = self.deconv7(x) 
        return x.squeeze(dim=1)

class Passive_heat_map_3D_intall(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(Passive_heat_map_3D_intall, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(31 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock3D(dim5, dim5, kernel_size=(4,5,4))
        self.deconv1_2 = ConvBlock3D(dim5, dim5)
        self.deconv2_1 = DeconvBlock3D(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock3D(dim4, dim4)
        self.deconv3_1 = DeconvBlock3D(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock3D(dim3, dim3)
        self.deconv4_1 = DeconvBlock3D(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock3D(dim2, dim2)
        self.deconv5_1 = DeconvBlock3D(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock3D(dim1, dim1)
        self.deconv6_1 = DeconvBlock3D(dim1, dim0, kernel_size=4, stride=(3,3,1), padding=1)
        self.deconv6_2 = ConvBlock(dim0, dim0)
        self.deconv7 = ConvBlock3D_Tanh(dim0, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        x = torch.unsqueeze(x,-1)
        #x = torch.unsqueeze(x,-1)
        #print('x after conv 8:',x.size())
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 4, 5, 4)
        #print('x after deconv 1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 4, 5, 4)
        x = self.deconv2_1(x) # (None, 256, 8, 10, 8) 
        x = self.deconv2_2(x) # (None, 256, 8, 10, 8)
        x = self.deconv3_1(x) # (None, 128, 16, 20, 16) 
        x = self.deconv3_2(x) # (None, 128, 16, 20, 16)
        x = self.deconv4_1(x) # (None, 64, 32, 40, 32) 
        x = self.deconv4_2(x) # (None, 64, 32, 40, 32)
        x = self.deconv5_1(x) # (None, 32, 64, 80, 64) 
        x = self.deconv5_2(x) # (None, 32, 64, 80, 64) 
        #print('size after decon5_2',x.size())
        x = self.deconv6_1(x) # (None, 16, 192, 240, 64) #
        x = self.deconv6_2(x) # (None, 16, 192, 240, 64)
        #print('size after decon6_2',x.size()) # nx ny nz 63 79 55  158 198 55
        x = F.pad(x, [-4, -5, -21, -21, -17, -17], mode="constant", value=0) 
        x = self.deconv7(x) 
        return x.squeeze(dim=1)


class Passive_heat_map_3D_intz_uq(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(Passive_heat_map_3D_intz_uq, self).__init__()
        self.convblock1 = Conv2DwithBNUQ(1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), dropout=False)
        self.convblock2_1 = Conv2DwithBNUQ(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dropout=False)
        self.convblock2_2 = Conv2DwithBNUQ(dim2, dim2, kernel_size=(3, 1), padding=(1, 0), dropout=False)
        self.convblock3_1 = Conv2DwithBNUQ(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dropout=False)
        self.convblock3_2 = Conv2DwithBNUQ(dim2, dim2, kernel_size=(3, 1), padding=(1, 0), dropout=False)
        self.convblock4_1 = Conv2DwithBNUQ(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), dropout=False)
        self.convblock4_2 = Conv2DwithBNUQ(dim3, dim3, kernel_size=(3, 1), padding=(1, 0), dropout=False)
        self.convblock5_1 = Conv2DwithBNUQ(dim3, dim3, stride=2, dropout=False)
        self.convblock5_2 = Conv2DwithBNUQ(dim3, dim3, dropout=False)
        self.convblock6_1 = Conv2DwithBNUQ(dim3, dim4, stride=2, dropout=False)
        self.convblock6_2 = Conv2DwithBNUQ(dim4, dim4, dropout=False)
        self.convblock7_1 = Conv2DwithBNUQ(dim4, dim4, stride=2, dropout=False)
        self.convblock7_2 = Conv2DwithBNUQ(dim4, dim4, dropout=False)
        self.convblock8 = Conv2DwithBNUQ(dim4, dim5, kernel_size=(8, ceil(31 * sample_spatial / 8)), padding=0, dropout=False)
        
        self.deconv1_1 = DeconvBlock3DUQ(dim5, dim5, kernel_size=(4,5,4), dropout=False)
        self.deconv1_2 = ConvBlock3DUQ(dim5, dim5, dropout=False)
        self.deconv2_1 = DeconvBlock3DUQ(dim5, dim4, kernel_size=4, stride=2, padding=1, dropout=False)
        self.deconv2_2 = ConvBlock3DUQ(dim4, dim4, dropout=False)
        self.deconv3_1 = DeconvBlock3DUQ(dim4, dim3, kernel_size=4, stride=2, padding=1, dropout=False)
        self.deconv3_2 = ConvBlock3DUQ(dim3, dim3, dropout=False)
        self.deconv4_1 = DeconvBlock3DUQ(dim3, dim2, kernel_size=4, stride=2, padding=1, dropout=False)
        self.deconv4_2 = ConvBlock3DUQ(dim2, dim2, dropout=False)
        self.deconv5_1 = DeconvBlock3DUQ(dim2, dim1, kernel_size=4, stride=2, padding=1, dropout=False)
        self.deconv5_2 = ConvBlock3DUQ(dim1, dim1, dropout=False)
        #self.deconv6_1 = DeconvBlock3D(dim1, dim0, kernel_size=4, stride=(4,5), padding=1)
        #self.deconv6_2 = ConvBlock(dim0, dim0)
        self.deconv7 = ConvBlock3D_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = F.dropout2d(x, p=0.2,training=True)
        x = self.convblock8(x) # (None, 512, 1, 1)
        x = F.dropout2d(x, p=0.2,training=True)
        #x = torch.unsqueeze(x,-1)
        x = torch.unsqueeze(x,-1)
        #print('x after conv 8:',x.size())
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 4, 5, 4)
        #print('x after deconv 1_1:',x.size())
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv1_2(x) # (None, 512, 4, 5, 4)
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv2_1(x) # (None, 256, 8, 10, 8) 
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv2_2(x) # (None, 256, 8, 10, 8)
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv3_1(x) # (None, 128, 16, 20, 16) 
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv3_2(x) # (None, 128, 16, 20, 16)
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv4_1(x) # (None, 64, 32, 40, 32) 
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv4_2(x) # (None, 64, 32, 40, 32)
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv5_1(x) # (None, 32, 64, 80, 64) 
        x = F.dropout2d(x, p=0.5,training=True)
        x = self.deconv5_2(x) # (None, 32, 64, 80, 64) 
        #print('size after decon5_2',x.size())
        #x = self.deconv6_1(x) # (None, 16, 320, 400) #
        #x = self.deconv6_2(x) # (None, 16, 320, 400)
        #print('size after decon6_2',x.size()) # nx ny nz 63 79 55
        x = F.dropout2d(x, p=0.5,training=True)
        x = F.pad(x, [-4, -5, -1, 0, -1, 0], mode="constant", value=0) 
        x = self.deconv7(x) 
        return x.squeeze(dim=1)

class Passive_heat_map_3D_intz_big(nn.Module):
    def __init__(self, dim0=32, dim1=64, dim2=128, dim3=256, dim4=512, dim5=1024, sample_spatial=1.0, **kwargs):
        super(Passive_heat_map_3D_intz_big, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim0, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim0, dim1, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim1, dim1, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(31 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock3D(dim5, dim4, kernel_size=(4,5,4))
        self.deconv1_2 = ConvBlock3D(dim4, dim4)
        self.deconv2_1 = DeconvBlock3D(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock3D(dim3, dim3)
        self.deconv3_1 = DeconvBlock3D(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock3D(dim2, dim2)
        self.deconv4_1 = DeconvBlock3D(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock3D(dim1, dim1)
        self.deconv5_1 = DeconvBlock3D(dim1, dim0, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock3D(dim0, dim0)
        #self.deconv6_1 = DeconvBlock3D(dim1, dim0, kernel_size=4, stride=(4,5), padding=1)
        #self.deconv6_2 = ConvBlock(dim0, dim0)
        self.deconv7 = ConvBlock3D_Tanh(dim0, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 7
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        x = torch.unsqueeze(x,-1)
        #print('x after conv 8:',x.size())
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 4, 5, 4)
        #print('x after deconv 1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 4, 5, 4)
        x = self.deconv2_1(x) # (None, 256, 8, 10, 8) 
        x = self.deconv2_2(x) # (None, 256, 8, 10, 8)
        x = self.deconv3_1(x) # (None, 128, 16, 20, 16) 
        x = self.deconv3_2(x) # (None, 128, 16, 20, 16)
        x = self.deconv4_1(x) # (None, 64, 32, 40, 32) 
        x = self.deconv4_2(x) # (None, 64, 32, 40, 32)
        x = self.deconv5_1(x) # (None, 32, 64, 80, 64) 
        x = self.deconv5_2(x) # (None, 32, 64, 80, 64) 
        #print('size after decon5_2',x.size())
        #x = self.deconv6_1(x) # (None, 16, 320, 400) #
        #x = self.deconv6_2(x) # (None, 16, 320, 400)
        #print('size after decon6_2',x.size()) # nx ny nz 63 79 55
        x = F.pad(x, [-4, -5, -1, 0, -1, 0], mode="constant", value=0) 
        x = self.deconv7(x) 
        return x.squeeze(dim=1)


class Passive_heat_map_3D_intz_deep(nn.Module):
    def __init__(self, dim0=16, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, dim6=1024, sample_spatial=1.0, **kwargs):
        super(Passive_heat_map_3D_intz_deep, self).__init__()
        self.convblock1 = Conv2DwithBN(1, dim0, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim0, dim1, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim1, dim1, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim2, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim5, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim5, dim5)
        self.convblock8_1 = Conv2DwithBN(dim5, dim6, stride=2)
        self.convblock8_2 = Conv2DwithBN(dim6, dim6)
        self.convblock9 = Conv2DwithBN(dim6, dim6, kernel_size=(4, 2), padding=0)
        
        self.deconv1_1 = DeconvBlock3D(dim6, dim6, kernel_size=(4,5,4))
        self.deconv1_2 = ConvBlock3D(dim6, dim5)
        self.deconv2_1 = DeconvBlock3D(dim5, dim5, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock3D(dim5, dim4)
        self.deconv3_1 = DeconvBlock3D(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock3D(dim3, dim3)
        self.deconv4_1 = DeconvBlock3D(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock3D(dim2, dim2)
        self.deconv5_1 = DeconvBlock3D(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock3D(dim1, dim1)
        #self.deconv6_1 = DeconvBlock3D(dim1, dim0, kernel_size=4, stride=(4,5), padding=1)
        #self.deconv6_2 = ConvBlock(dim0, dim0)
        self.deconv7 = ConvBlock3D_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8_1(x) # (None, 256, 4, 5) 
        x = self.convblock8_2(x) # (None, 256, 4, 5)
        #print(x.size())
        x = self.convblock9(x) # (None, 512, 1, 1)
        x = torch.unsqueeze(x,-1)
        #print('x after conv 8:',x.size())
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 4, 5, 4)
        #print('x after deconv 1_1:',x.size())
        x = self.deconv1_2(x) # (None, 512, 4, 5, 4)
        x = self.deconv2_1(x) # (None, 256, 8, 10, 8) 
        x = self.deconv2_2(x) # (None, 256, 8, 10, 8)
        x = self.deconv3_1(x) # (None, 128, 16, 20, 16) 
        x = self.deconv3_2(x) # (None, 128, 16, 20, 16)
        x = self.deconv4_1(x) # (None, 64, 32, 40, 32) 
        x = self.deconv4_2(x) # (None, 64, 32, 40, 32)
        x = self.deconv5_1(x) # (None, 32, 64, 80, 64) 
        x = self.deconv5_2(x) # (None, 32, 64, 80, 64) 
        #print('size after decon5_2',x.size())
        #x = self.deconv6_1(x) # (None, 16, 320, 400) #
        #x = self.deconv6_2(x) # (None, 16, 320, 400)
        #print('size after decon6_2',x.size()) # nx ny nz 63 79 55
        x = F.pad(x, [-4, -5, -1, 0, -1, 0], mode="constant", value=0) 
        x = self.deconv7(x) 
        return x.squeeze(dim=1)



model_dict = {
    'FCN4_No_BN': FCN4_No_BN,
    'FCN4_BN': FCN4_BN,
    'FCN4_Deep': FCN4_Deep,
    'FCN4_Deep_Resize': FCN4_Deep_Resize,
    'FCN4_Deep_2': FCN4_Deep_2,
    'SeisT_Decoder': SeisT_Decoder,
    'SeisT_Decoder_Resize': SeisT_Decoder_Resize,
    'ViT_Decoder': ViT_Decoder,
    'ViT_Decoder_Resize': ViT_Decoder_Resize,
    'MLP_Mixer_Decoder': MLP_Mixer_Decoder,
    'MLP_Mixer_Decoder_Resize': MLP_Mixer_Decoder_Resize,
    'FCN4_Deep_Resize_2': FCN4_Deep_Resize_2,
    'FCN4_V2S_Deep': FCN4_V2S_Deep,
    'FCN4_V2S_Deep_2': FCN4_V2S_Deep_2,
    'FCN4_Deep_3': FCN4_Deep_3,
    'FCN4_Deep_4': FCN4_Deep_4,
    'FCN4_Salt': FCN4_Salt,
    'FCN4_Salt_Resize': FCN4_Salt_Resize,
    'Discriminator': Discriminator,
    'InversionNet': InversionNet,
    'HPGNN': HPGNN,
    'PDE_Regress': PDE_Regress,
    'PDE_Regress_CO2': PDE_Regress_CO2,
    'PDE_Regress_CO2_UNet': PDE_Regress_CO2_UNet,
    'PDE_Regress_CO2_UNet_dv_all_labels_diff':PDE_Regress_CO2_UNet_dv_all_labels_diff,
    'PDE_Regress_CO2_UNet_dv_all_labels':PDE_Regress_CO2_UNet_dv_all_labels,
    'PDE_Regress_CO2_UNet_dv_all_labels_diff_o3':PDE_Regress_CO2_UNet_dv_all_labels_diff_o3,
    'PDE_Regress_CO2_UNet_dv_all_labels_o3':PDE_Regress_CO2_UNet_dv_all_labels_o3,
    'PDE_Regress_CO2_UNet_dv_all_labels_diff_i30o3':PDE_Regress_CO2_UNet_dv_all_labels_diff_i30o3,
    'PDE_Regress_CO2_UNet_dv_all_labels_i10o3':PDE_Regress_CO2_UNet_dv_all_labels_i10o3,
    'FCN4_Deep_Resize_2_mlreal':FCN4_Deep_Resize_2_mlreal,
    'FCN4_Deep_Resize_2_mlreal_o3':FCN4_Deep_Resize_2_mlreal_o3,
    'FCN4_Deep_Resize_2_mlreal_large':FCN4_Deep_Resize_2_mlreal_large,
    'Passive_heat_map':Passive_heat_map,
    'Passive_heat_map_3D':Passive_heat_map_3D,
    'Passive_heat_map_3D_intz':Passive_heat_map_3D_intz,
    'Passive_heat_map_3D_intall':Passive_heat_map_3D_intall,
    'Passive_heat_map_3D_intz_uq':Passive_heat_map_3D_intz_uq,
    'Passive_heat_map_3D_intz_big':Passive_heat_map_3D_intz_big,
    'Passive_heat_map_3D_intz_deep':Passive_heat_map_3D_intz_deep
}

if __name__ == '__main__':
    device = torch.device('cpu')
    # model = MLP_Mixer_Decoder_Resize() # 35594827
    # model = MLP_Mixer_Decoder()        # 41007691
    # model = ViT_Decoder_Resize()       # 63177315
    # model = ViT_Decoder()              # 68590179
    # model = FCN4_Salt_Resize()         # 11737379
    # model = FCN4_Salt()                # 13742371
    ratio = 0.1
    model = FCN4_Deep_Resize_2(ratio=ratio)       # 18996259
    # model = FCN4_V2S_Deep_2()          # 83531903, 65706111
    # model = FCN4_V2S_Deep()            # 22972031
    # model = SeisT_Decoder()            # 68590179
    # model = SeisT_Decoder_Resize()     # 63177315
    # model = FCN4_No_BN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: %d' % total_params)
    # x= torch.rand((3, 9, 401, 301))
    # x = torch.rand((3, 5, 600, 60))
    # x = torch.rand((3, 1, 70, 70))
    x = torch.rand((3, 2, 141, 401))
    y = model(x)
    print(y.shape)


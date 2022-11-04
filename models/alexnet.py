# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d

from args import args
from .builder import Builder

def conv3x3(in_planes, out_planes, builder, stride=1):
    return builder.conv3x3(in_planes, out_planes, stride=stride)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, builder, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, builder, stride)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = conv3x3(planes, planes, builder)
        self.bn2 = builder.batchnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class RN(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf=128):  # originally nf was 20, not 128
        super(RN, self).__init__()
        self.in_planes = nf

        builder = Builder()

        self.conv1 = conv3x3(3, nf * 1, builder)
        self.bn1 = builder.batchnorm(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], builder, stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], builder, stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], builder, stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], builder, stride=2)
        self.linear = builder.conv1x1(nf * 8 * block.expansion, num_classes, last_layer=True)

    def _make_layer(self, block, planes, num_blocks, builder, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, builder, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) # (, 3, 32, 32) -> (, 20, 32, 32)
        out = self.layer1(out) # (, 20, 32, 32) -> (, 20, 32, 32) -> (, 20, 32, 32) -> (, 20, 32, 32) -> (, 20, 32, 32)
        out = self.layer2(out) # (, 20, 32, 32) -> (, 40, 16, 16) -> (, 40, 16, 16) -> (, 40, 16, 16) -> (, 40, 16, 16)
        out = self.layer3(out) # (, 40, 16, 16) -> (, 80, 8, 8) -> (, 80, 8, 8) -> (, 80, 8, 8) -> (, 80, 8, 8)
        out = self.layer4(out) # (, 40, 8, 8) -> (, 160, 4, 4) -> (, 160, 4, 4) -> (, 160, 4, 4) -> (, 160, 4, 4)
        out = avg_pool2d(out, 4) # (, 160, 4, 4) -> (, 160, 1, 1)
        out = self.linear(out) # 160 -> num cls per task, final output shape is (, num_cls_per_task, 1, 1)
        out = out.view(out.size(0), -1)
        return out


def GEMResNet18():
    return RN(BasicBlock, [2, 2, 2, 2], args.output_size, nf=int(args.width_mult * 128)) # originally it was 20, not 128

class Net(nn.Module):
    def __init__(self, num_classes):  # originally nf was 20, not 128
        super(Net, self).__init__()
        builder = Builder()

        self.conv1 = conv3x3(3, 64, builder)
        self.conv2 = conv3x3(64, 128, builder)
        self.conv3 = conv3x3(128, 256, builder)

        self.fc1 = builder.conv1x1(256, 800, last_layer=False)
        self.fc = builder.conv1x1(800, num_classes, last_layer=True)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        bsz = x.size(0)
        out = self.maxpool(self.drop1(self.conv1(x.view(bsz, 3, 28, 28)))) # torch.Size([3, 64, 13, 13])
        out = self.maxpool(self.drop1(self.relu(self.conv2(out)))) # torch.Size([3, 128, 5, 5])
        out = self.maxpool(self.drop2(self.relu(self.conv3(out)))) # torch.Size([3, 256, 2, 2])

        out = avg_pool2d(out, 2)
        out = self.drop2(self.relu(self.fc1(out)))
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        return out

def Alexnet():
    return Net(args.output_size)




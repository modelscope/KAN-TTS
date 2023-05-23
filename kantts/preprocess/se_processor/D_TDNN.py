from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from .layers import (DenseLayer, DenseTDNNBlock, StatsPool, TDNNLayer, SEDenseTDNNBlock,
                     TransitLayer)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class CNN_Head(nn.Module):
    def __init__(self,
                block=BasicBlock,
                num_blocks=[2, 2],
                m_channels=32,
                feat_dim=80):
        super(CNN_Head, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels =  m_channels * (feat_dim // 8)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = out.reshape(out.shape[0], out.shape[1]*out.shape[2], out.shape[3])
        return out

class DTDNN(nn.Module):
    def __init__(self,
                 feat_dim=80,
                 embedding_size=192,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True):
        super(DTDNN, self).__init__()

        self.head = CNN_Head()
        feat_dim = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(feat_dim,
                           init_channels,
                           5,
                           stride=2,
                           dilation=1,
                           padding=-1,
                           config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size,
                dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 3))):
            block = SEDenseTDNNBlock(num_layers=num_layers,
                                   in_channels=channels,
                                   out_channels=growth_rate,
                                   bn_channels=bn_size * growth_rate,
                                   kernel_size=kernel_size,
                                   dilation=dilation,
                                   config_str=config_str,
                                   memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(channels,
                             channels // 2,
                             bias=False,
                             config_str=config_str))
            channels //= 2

        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module(
            'dense',
            DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector.tdnn(x)
        
        x = self.xvector.block1(x)
        x = self.xvector.transit1(x)

        x = self.xvector.block2(x)
        x = self.xvector.transit2(x)

        x = self.xvector.block3(x)
        x = self.xvector.transit3(x)
        x = self.relu(self.bn(x))

        x = self.xvector.stats(x)
        x = self.xvector.dense(x)
        return x


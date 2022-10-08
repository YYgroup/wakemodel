"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import pdb

import torch
import torch.nn as nn
from .bam import *
from .cbam import *

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        print('CNN model: resnet')
        num_classes = 50 # 输出只有1个值
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.fc2(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



class ResNetPlate(nn.Module):
    """the cnn model for predicting the force"""

    def __init__(self, block, num_block, add_aoa=False, add_fre=False,
                 att_type = None, spatial=False, channel=False, add_fc_relu=False, add_fc_sigmoid=False):
        super().__init__()

        print('CNN model: ResNetPlate')
        print('add_aoa:', add_aoa)
        print('add_fre:', add_fre)
        print('att_type:', att_type)
        print('spatial attention:', spatial)
        print('channel attention:', channel)
        print('add_fc_relu:', add_fc_relu)
        print('add_fc_sigmoid:', add_fc_sigmoid)
        mid_dimension = 50 
        self.add_aoa = add_aoa
        self.add_fre = add_fre
        self.in_channels = 64
        self.add_fc_relu = add_fc_relu
        self.add_fc_sigmoid = add_fc_sigmoid

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)

        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, mid_dimension)
        self.fc_aoa = nn.Linear(1, mid_dimension)  # 全连接层中加入平板攻角信息
        self.fc_wake_dis = nn.Linear(1, mid_dimension)
        self.fc_fre = nn.Linear(1, mid_dimension)
        if self.add_fc_relu:
            self.fc_relu = nn.ReLU(inplace=True)
        elif self.add_fc_sigmoid:
            self.fc_sigmoid = nn.Sigmoid()

        self.fc_force = nn.Linear(mid_dimension, 1)

        if att_type == 'BAM':
            self.bam2 = BAM(64 * block.expansion, channel = channel, spatial=spatial)
            self.bam3 = BAM(128 * block.expansion, channel = channel, spatial=spatial)
            self.bam4 = BAM(256 * block.expansion, channel = channel, spatial=spatial)
        else:
            self.bam2, self.bam3, self.bam4 = None, None, None

        # 初始化权重参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, aoa=None, wake_dis=None, fre=None):
        output = self.conv1(x)
        output = self.conv2_x(output)
        if self.bam2 is not None:
            output, att = self.bam2(output)

        output = self.conv3_x(output)
        if self.bam3 is not None:
            output, att = self.bam3(output)

        output = self.conv4_x(output)
        if self.bam4 is not None:
            output, att = self.bam4(output)

        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output1 = output.view(output.size(0), -1)

        output = self.fc(output1)

        if self.add_aoa and aoa is not None:  # 插入攻角的先验信息
            aoa = aoa.unsqueeze(1)  # 此时aoa.shape: torch.Size([64, 1])
            output_aoa = self.fc_aoa(aoa)
            output = output+ output_aoa

        if self.add_fre and fre is not None:
            fre = fre.unsqueeze(1)
            output_fre = self.fc_fre(fre)
            output = output+output_fre

        if self.add_fc_relu:
            output = self.fc_relu(output)
        elif self.add_fc_sigmoid:
            output = self.fc_sigmoid(output)

        output = self.fc_force(output)
        return output

"""
更小的模型resnet9
"""
def resnet9():
    print('resnet9')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1])

def resnet9_fc_relu():
    print('resnet9')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_fc_relu=True)

def resnet9_fc_sigmoid():
    print('resnet9')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_fc_sigmoid=True)

def resnet9_aoa():
    print('resnet9 with aoa')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_aoa=True)

def resnet9_aoa_fc_relu():
    print('resnet9 with aoa')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_aoa=True, add_fc_relu=True)

def resnet9_aoa_fc_sigmoid():
    print('resnet9 with aoa')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_aoa=True, add_fc_sigmoid=True)

def resnet9_fre():
    print('resnet9 with fre')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_fre=True)

def resnet9_aoa_fre():
    print('resnet9 with aoa and fre')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_aoa=True, add_fre=True)

def resnet9_aoa_spatt():
    print('resnet9 with aoa, att_type BAM, spatial attention')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_aoa=True, att_type='BAM', spatial=True)

def resnet9_spatt():
    print('resnet9  att_type BAM, spatial attention')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], att_type='BAM', spatial=True)

def resnet9_fre_spatt():
    print('resnet9 with fre att')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_fre=True, att_type='BAM', spatial=True)

def resnet9_aoa_fre_spatt():
    print('resnet9 with aoa and fre att_type BAM, spatial attention')
    return ResNetPlate(BasicBlock, [1, 1, 1, 1], add_aoa=True, add_fre=True, att_type='BAM',
                       spatial=True)


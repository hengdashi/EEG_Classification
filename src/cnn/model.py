'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
# third party libs
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv1d(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm1d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv1d(planes, planes, stride)
    self.bn2 = nn.BatchNorm1d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=4):
    self.inplanes = 64

    super(ResNet, self).__init__()
    self.conv1 = nn.Conv1d(22, 64, kernel_size=7, stride=2, padding=3,
                            bias=True)
    self.bn1 = nn.BatchNorm1d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool1d(7, stride=1)
    self.fc = nn.Linear(124928, num_classes)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    # block.expansion is 1
    # when stride is not 1, and planes is not 64 (aka. self.inplanes), init a downsample
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv1d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=True),
          nn.BatchNorm1d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    # (N, 22, 1000)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    # (N, 22, 500)
    x = self.maxpool(x)
    # (N, 22, 250)

    x = self.layer1(x)
    # (N, 64, 250)
    x = self.layer2(x)
    # (N, 128, 250)
    x = self.layer3(x)
    # (N, 256, 250)
    x = self.layer4(x)
    # (N, 512, 250)

    x = self.avgpool(x)
    # (N, 512, 244)
    x = torch.flatten(x, 1)
    # (N, 124968)
    x = self.fc(x)

    return x



class DeepConvNet(nn.Module):
  def __init__(self, args, num_classes=4):
    super(DeepConvNet, self).__init__()

    self.n_features = args['common']['n_features']

    kernel_size = args['cnn']['kernel']
    pool_size = args['cnn']['pool']
    dropout = args['cnn']['dropout']

    # conv 1
    self.conv11   = nn.Conv2d(1, 25, kernel_size=(1, kernel_size))
    self.bn11     = nn.BatchNorm2d(25)
    self.elu11    = nn.ELU()

    self.conv12   = nn.Conv2d(25, 25, kernel_size=(self.n_features, 1))
    self.bn12     = nn.BatchNorm2d(25)
    self.elu12    = nn.ELU()

    self.maxpool1 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=3)
    self.dropout1 = nn.Dropout(dropout)

    # conv 2
    self.conv2    = nn.Conv2d(25, 50, kernel_size=(1, kernel_size))
    self.bn2      = nn.BatchNorm2d(50)
    self.elu2     = nn.ELU()

    self.maxpool2 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=3)
    self.dropout2 = nn.Dropout(dropout)

    # conv 3
    self.conv3    = nn.Conv2d(50, 100, kernel_size=(1, kernel_size))
    self.bn3      = nn.BatchNorm2d(100)
    self.elu3     = nn.ELU()
    self.maxpool3 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=3)
    self.dropout3 = nn.Dropout(dropout)

    # conv 4
    self.conv4    = nn.Conv2d(100, 200, kernel_size=(1, kernel_size))
    self.bn4      = nn.BatchNorm2d(200)
    self.elu4     = nn.ELU()
    self.maxpool4 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=3)
    self.dropout4 = nn.Dropout(dropout)

    # fc layer
    if args['common']['scale'] == 2:
      self.fc1    = nn.Linear(2600, num_classes)
    else:
      self.fc1    = nn.Linear(200, num_classes)


  def forward(self, X):
    batch_size = X.size(0)

    # conv 1
    output = X.view(batch_size, 1, self.n_features, -1)
    # print(output.shape)
    output = self.conv11(output)
    output = self.bn11(output)
    output = self.elu11(output)
    # print(output.shape)

    output = self.conv12(output)
    output = self.bn12(output)
    output = self.elu12(output)

    output = self.maxpool1(output)
    output = self.dropout1(output)

    # conv 2
    output = self.conv2(output)
    output = self.bn2(output)
    output = self.elu2(output)

    output = self.maxpool2(output)
    output = self.dropout2(output)

    # conv 3
    output = self.conv3(output)
    output = self.bn3(output)
    output = self.elu3(output)

    output = self.maxpool3(output)
    output = self.dropout3(output)

    # conv 4
    output = self.conv4(output)
    output = self.bn4(output)
    output = self.elu4(output)

    output = self.maxpool4(output)
    output = self.dropout4(output)
    # print(output.shape)

    # reshape to feed in fc
    output = output.view(batch_size, -1)

    return self.fc1(output)

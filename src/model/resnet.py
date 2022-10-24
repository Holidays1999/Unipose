# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os

import numpy as np
from mindspore import Tensor, dtype as mstype
from mindspore import nn, context

if os.getenv("DEVICE_TARGET", "GPU") == "GPU" or int(os.getenv("DEVICE_NUM")) == 1:
    BatchNorm2d = nn.BatchNorm2d
elif os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    raise ValueError(f"Model doesn't support devide_num = {int(os.getenv('DEVICE_NUM'))} "
                     f"and device_target = {os.getenv('DEVICE_TARGET')}")


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, pad_mode='pad')
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, pad_mode='pad')
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, pad_mode='pad')
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(self, block, layers, output_stride):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], )
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], )

        # self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.SequentialCell(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion, pad_mode='pad',
                          kernel_size=1, stride=stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i] * dilation))

        return nn.SequentialCell(*layers)

    def construct(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat


def ResNet101(output_stride):
    """Constructs a ResNet-101 model.
    Args:
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride)

    return model


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = ResNet101(output_stride=8)

    input = Tensor(np.random.randn(1, 3, 512, 512), dtype=mstype.float32)

    output, low_level_feat = net(input)
    print('output: ', output.shape)  # (1, 2048, 64, 64)
    print('low_level_feat: ', low_level_feat.shape)  # (1, 256, 127, 127)
    params = 0.
    num = 0.
    for name, param in net.parameters_and_names():
        params += np.prod(param.shape)
        print(name, param.shape)
        num += 1
    print(params, num) # 42605504.0 520.0

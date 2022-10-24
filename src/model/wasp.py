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
from mindspore import context, nn, ops

from src.model.weight_init import init_weights

if os.getenv("DEVICE_TARGET", "GPU") == "GPU" or int(os.getenv("DEVICE_NUM")) == 1:
    BatchNorm2d = nn.BatchNorm2d
elif os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    raise ValueError(f"Model doesn't support devide_num = {int(os.getenv('DEVICE_NUM'))} "
                     f"and device_target = {os.getenv('DEVICE_TARGET')}")


class GlobalPool2D(nn.Cell):
    def __init__(self):
        """Initialize AdaptiveAvgPool2D."""
        super(GlobalPool2D, self).__init__()

    def construct(self, input):
        x = ops.ReduceMean(True)(input, (2, 3))
        return x


class AtrousModule(nn.Cell):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(AtrousModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, pad_mode='pad',
                                     stride=1, padding=padding, dilation=dilation)
        self.bn = BatchNorm2d(planes)
        self.act = nn.ReLU()
        init_weights(self)

    def construct(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.act(x)


class WASP(nn.Cell):
    def __init__(self, output_stride):
        super(WASP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [48, 24, 12, 6]
        elif output_stride == 8:
            dilations = [48, 36, 24, 12]
        else:
            raise NotImplementedError

        self.aspp1 = AtrousModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = AtrousModule(256, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = AtrousModule(256, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = AtrousModule(256, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.SequentialCell(
            GlobalPool2D(),
            nn.Conv2d(inplanes, 256, 1, stride=1),
            BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(1280, 256, 1)
        self.conv2 = nn.Conv2d(256, 256, 1, )
        self.bn1 = BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=1 - 0.5)
        init_weights(self)

    def construct(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x5 = self.global_avg_pool(x)
        x5 = ops.ResizeBilinear(size=x4.shape[2:], align_corners=True)(x5)
        x = ops.Concat(axis=1)((x1, x2, x3, x4, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


def build_wasp(output_stride):
    return WASP(output_stride)


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = build_wasp(output_stride=16)
    inputs = Tensor(np.random.randn(1, 2048, 64, 64), mstype.float32)
    output = net(inputs)
    print(f'output: {output.shape}')

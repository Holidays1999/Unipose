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

from mindspore import nn, ops

from src.model.weight_init import init_weights

if os.getenv("DEVICE_TARGET", "GPU") == "GPU" or int(os.getenv("DEVICE_NUM")) == 1:
    BatchNorm2d = nn.BatchNorm2d
elif os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    raise ValueError(f"Model doesn't support devide_num = {int(os.getenv('DEVICE_NUM'))} "
                     f"and device_target = {os.getenv('DEVICE_TARGET')}")


class Decoder(nn.Cell):
    def __init__(self, num_classes, backbone):
        super(Decoder, self).__init__()
        assert backbone == 'ResNet101'
        low_level_inplanes = 256

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1)
        self.bn1 = BatchNorm2d(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.SequentialCell(
            nn.Conv2d(304, 256, kernel_size=3),
            BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Conv2d(256, 256, kernel_size=3),
            BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(keep_prob=1 - 0.1),
            nn.Conv2d(256, num_classes + 1, kernel_size=1)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.init_weights()

    def construct(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        low_level_feat = self.maxpool(low_level_feat)

        x = ops.ResizeBilinear(size=low_level_feat.shape[2:], align_corners=True)(x)
        x = ops.Concat(axis=1)((x, low_level_feat))
        x = self.last_conv(x)

        return x

    def init_weights(self):
        for name, cell in self.cells_and_names():
            init_weights(cell)


def build_decoder(num_classes, backbone):
    return Decoder(num_classes, backbone)

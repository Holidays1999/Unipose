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
import numpy as np
from mindspore import Tensor, dtype as mstype
from mindspore import nn, context
from mindspore import ops

from src.model.decoder import build_decoder
from src.model.resnet import ResNet101
from src.model.wasp import build_wasp
from src.model.weight_init import init_weights


def build_backbone(backbone, output_stride):
    if backbone == 'ResNet101':
        return ResNet101(output_stride)
    else:
        raise NotImplementedError


class Unipose(nn.Cell):
    def __init__(self, cfg, backbone='resnet', output_stride=16, num_classes=21, stride=8):
        super(Unipose, self).__init__()
        self.stride = stride
        self.heatmap_h = cfg.heatmap_h
        self.heatmap_w = cfg.heatmap_w

        self.num_classes = num_classes

        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8)

        self.backbone = build_backbone(backbone, output_stride)

        self.wasp = build_wasp(output_stride)
        self.decoder = build_decoder(num_classes, backbone)
        init_weights(self)

    def construct(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)
        if self.stride != 8:
            x = ops.ResizeBilinear(size=(self.heatmap_h, self.heatmap_w), align_corners=True)(x)
        return x


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    model = Unipose(backbone='resnet', stride=8)
    inputs = Tensor(np.random.randn(1, 3, 184, 92), dtype=mstype.float32)
    output = model(inputs)
    print(output.shape)  # (1, 22, 46, 23)

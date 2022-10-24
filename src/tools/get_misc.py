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

from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.engine.train_engine import TrainOneStepWithGlobalNormClip


def get_train_one_step(args, net_with_loss, optimizer):
    """get_train_one_step cell"""
    if args.is_dynamic_loss_scale:
        print(f"=> Using DynamicLossScaleUpdateCell")
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 10, scale_factor=2,
                                                                    scale_window=2000)
    else:
        print(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)
    net_with_loss = TrainOneStepWithGlobalNormClip(net_with_loss, optimizer, global_norm=args.clip_global_norm_value,
                                                   scale_sense=scale_sense)

    return net_with_loss


def get_pretrained(args, model):
    """"Load pretrained weights if args.pretrained is given"""
    if args.run_modelarts:
        # print('Syncing data.')
        # local_data_path = '/cache/weight/model.ckpt'
        # sync_data(args.pretrained, local_data_path, threads=128)
        args.pretrained = os.path.join(args.data_url, "resnet101_torch.ckpt")
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        load_param_into_net(model, param_dict)
    elif os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        load_param_into_net(model, param_dict)
    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

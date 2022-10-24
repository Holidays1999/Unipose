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
import random

import numpy as np
from mindspore import Model, context, nn
from mindspore.common import set_seed

from src.args import args
from src.data.dataset import CreateDatasetLSP
from src.model.unipose import Unipose
from src.tools.amp import cast_amp
from src.tools.eval_metric import PCKMetric
from src.tools.get_misc import get_train_one_step, get_pretrained
from src.tools.loss import JointsMSELoss
from src.tools.optim import get_optimizer
from src.tools.prepare_misc import prepare_context, prepare_callbacks


def main(args):
    # prepare context
    set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    args.rank = prepare_context(args)

    # create dataset
    train_dataset, val_dataset = CreateDatasetLSP(args=args)

    # create model
    model = Unipose(cfg=args, num_classes=args.num_classes, backbone=args.backbone, output_stride=16,
                    stride=args.stride)

    # precare mix precision
    cast_amp(args=args, net=model)

    # create optimizer
    optimizer = get_optimizer(args=args, batch_num=train_dataset.get_dataset_size(), model=model)

    # create net with loss
    loss_fn = JointsMSELoss()
    net_with_loss = nn.WithLossCell(model, loss_fn=loss_fn)

    # get pretrained weight
    get_pretrained(args=args, model=net_with_loss)

    # create train one step cell
    train_one_step_cell = get_train_one_step(args, net_with_loss, optimizer)

    # create eval cell
    eval_network = nn.WithEvalCell(network=model, loss_fn=loss_fn, add_cast_fp32=args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]

    # create Model
    model = Model(network=train_one_step_cell,
                  metrics={"loss": nn.Loss(), "PCK": PCKMetric(num_classes=args.num_classes, threshold_PCK=args.th_pck,
                                                               threshold_PCKh=args.th_pckh)},
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    # prepare callbacks
    args.ckpt_save_dir = "./ckpt_" + str(args.rank)
    if args.run_modelarts:
        args.ckpt_save_dir = "/cache/ckpt_" + str(args.rank)
    callback_list = prepare_callbacks(args=args, train_dataset=train_dataset, val_dataset=val_dataset, model=model)

    print("begin train")
    model.train(args.epochs, train_dataset, callbacks=callback_list)
    print("train success")
    if args.run_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=args.ckpt_save_dir,
                               dst_url=os.path.join(args.train_url, "ckpt_" + str(args.rank)))


if __name__ == '__main__':
    main(args=args)

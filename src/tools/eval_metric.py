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
from mindspore import nn

from src.engine.eval_engine import accuracy
from src.engine.eval_engine import printAccuracies


class PCKMetric(nn.Metric):
    def __init__(self, num_classes, threshold_PCK=0.2, threshold_PCKh=0.5):
        super(PCKMetric, self).__init__()
        self.PCK = np.zeros(num_classes + 1)
        self.PCKh = np.zeros(num_classes + 1)
        self.count = np.zeros(num_classes + 1)
        self.num_classes = num_classes
        self.threshold_PCK = threshold_PCK
        self.threshold_PCKh = threshold_PCKh
        self._total_num = 0

        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self.PCK = np.zeros(self.num_classes + 1)
        self.PCKh = np.zeros(self.num_classes + 1)
        self.count = np.zeros(self.num_classes + 1)
        self._total_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError("For 'Accuracy.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}".format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        acc, acc_PCK, acc_PCKh, cnt, pred, visible = accuracy(y_pred, y, thr_PCK=self.threshold_PCK,
                                                              thr_PCKh=self.threshold_PCKh)
        self.PCK[0] = (self.PCK[0] * self._total_num + acc_PCK[0]) / (self._total_num + 1)
        self.PCKh[0] = (self.PCKh[0] * self._total_num + acc_PCKh[0]) / (self._total_num + 1)

        for j in range(1, self.num_classes + 1):
            if visible[j] == 1:
                self.PCK[j] = (self.PCK[j] * self.count[j] + acc_PCK[j]) / (self.count[j] + 1)
                self.PCKh[j] = (self.PCKh[j] * self.count[j] + acc_PCKh[j]) / (self.count[j] + 1)
                self.count[j] += 1

        self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError("The 'PCKMetric' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")
        mPCK = self.PCK[1:].sum() / self.num_classes
        mPCKh = self.PCKh[1:].sum() / self.num_classes
        printAccuracies(mPCKh, self.PCKh, mPCK, self.PCK, self.threshold_PCK, self.threshold_PCKh)
        return mPCK, mPCKh

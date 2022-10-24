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


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    # dists = np.zeros((preds.shape[1], preds.shape[0]))
    mask = 1 - (target[:, :, 0] > 1) * (target[:, :, 1] > 1)
    mask = np.squeeze(mask)
    normed_preds = preds / np.array(normalize)[:, np.newaxis]
    normed_targets = target / np.array(normalize)[:, np.newaxis]
    dists = np.linalg.norm(normed_preds - normed_targets, axis=-1)
    dists[mask] = -1
    return dists


def dist_acc(dists, threshold=0.5):
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()

    if num_dist_cal > 0:
        return np.less(dists[dist_cal], threshold).sum() * 1.0 / num_dist_cal
    else:
        return -1


def get_max_preds(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask

    return preds, maxvals


def accuracy(output, target, thr_PCK, thr_PCKh, hm_type='gaussian'):
    idx = list(range(output.shape[1]))

    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    else:
        raise ValueError
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx)))
    avg_acc = 0
    cnt = 0
    visible = np.zeros((len(idx)))
    for i in range(len(idx)):
        inputs = dists[:, idx[i]]
        result = dist_acc(inputs)
        acc[i] = result
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1
            visible[i] = 1
        else:
            acc[i] = 0

    avg_acc = avg_acc / cnt if cnt != 0 else 0

    if cnt != 0:
        acc[0] = avg_acc

    # PCKh
    PCKh = np.zeros((len(idx)))
    avg_PCKh = 0

    headLength = np.linalg.norm(target[0, 14, :] - target[0, 13, :])

    for i in range(len(idx)):
        PCKh[i] = dist_acc(dists[:, idx[i]], thr_PCKh * headLength)
        if PCKh[i] >= 0:
            avg_PCKh = avg_PCKh + PCKh[i]
        else:
            PCKh[i] = 0

    avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

    if cnt != 0:
        PCKh[0] = avg_PCKh

    # PCK
    PCK = np.zeros((len(idx)))
    avg_PCK = 0

    pelvis = [(target[0, 3, 0] + target[0, 4, 0]) / 2, (target[0, 3, 1] + target[0, 4, 1]) / 2]
    torso = np.linalg.norm(target[0, 13, :] - pelvis)

    for i in range(len(idx)):
        PCK[i] = dist_acc(dists[:, idx[i]], thr_PCK * torso)
        if PCK[i] >= 0:
            avg_PCK = avg_PCK + PCK[i]
        else:
            PCK[i] = 0

    avg_PCK = avg_PCK / cnt if cnt != 0 else 0

    if cnt != 0:
        PCK[0] = avg_PCK

    return acc, PCK, PCKh, cnt, pred, visible


def printAccuracies(mPCKh, PCKh, mPCK, PCK, thr_PCK, thr_PCKh):
    "show something"
    for index in range(len(PCKh)):
        PCKh[index] = PCKh[index] * 100
    for index in range(len(PCK)):
        PCK[index] = PCK[index] * 100
    content = f"mPCK@{thr_PCK}:  {mPCK * 100:.2f}%\n"
    content += f"PCK@0.2 s: Void={PCK[0]:.2f}%\n" \
               f"Right Ankle={PCK[1]:.2f}%\n" \
               f"Right Knee={PCK[2]:.2f}%\n" \
               f"Right Hip={PCK[3]:.2f}%\n" \
               f"Left Hip={PCK[4]:.2f}%\n"
    content += f"Left Knee={PCK[5]:.2f}%\n" \
               f"Left Ankle={PCK[6]:.2f}%\n" \
               f"Right Wrist={PCK[7]:.2f}%\n" \
               f"Right Elbow={PCK[8]:.2f}%\n" \
               f"Right Shoulder={PCK[9]:.2f}%\n"
    content += f"Left Shoulder={PCK[10]:.2f}%\n" \
               f"Left Elbow={PCK[11]:.2f}%\n" \
               f"Left Wrist={PCK[12]:.2f}%\n" \
               f"Neck={PCK[13]:.2f}%\n" \
               f"Head Top={PCK[14]:.2f}%\n"
    content += f"mPCKh@{thr_PCKh}: {mPCKh * 100}%\n"
    content += f"PCKh@{thr_PCKh} s: Void={PCKh[0]:.2f}%\n" \
               f"Right Ankle={PCKh[1]:.2f}%\n" \
               f"Right Knee={PCKh[2]:.2f}%\n" \
               f"Right Hip={PCKh[3]:.2f}%\n" \
               f"Left Hip={PCKh[4]:.2f}%\n"
    content += f"Left Knee={PCKh[5]:.2f}%\n" \
               f"Left Ankle={PCKh[6]:.2f}%\n" \
               f"Right Wrist={PCKh[7]:.2f}%\n" \
               f"Right Elbow={PCKh[8]:.2f}%\n" \
               f"Right Shoulder={PCKh[9]:.2f}%"
    content += f"Left Shoulder={PCKh[10]:.2f}%\n" \
               f"Left Elbow={PCKh[11]:.2f}%\n" \
               f"Left Wrist={PCKh[12]:.2f}%\n" \
               f"Neck={PCKh[13]:.2f}%\n" \
               f"Head Top={PCKh[14]:.2f}%"
    print(content)

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
import glob
import os

import cv2
import mindspore.dataset as ds
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image

import src.data.transforms as transforms
from src.tools.moxing_adapter import sync_data


def read_data_file(root_dir):
    """get train or val images
        return: image list: train or val images list
    """
    print(root_dir)
    image_list = glob.glob(os.path.join(root_dir, "images", '*.jpg'))
    for idx in range(len(image_list)):
        image_list[idx] = image_list[idx].replace('\\', '/')
    image_list = sorted(image_list)
    return image_list


def read_mat_file(mode, root_dir, img_list):
    """
        get the groundtruth

        mode (str): 'joints_train' or 'joints_test'
        return: three list: key_points list , centers list and scales list

        Notice:
            lsp_dataset differ from lspet dataset
    """
    assert mode in ["lsp", "lspet"]
    if mode == "lspet":
        mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
        lms = mat_arr.transpose([2, 1, 0])
        kpts = mat_arr.transpose([2, 0, 1]).tolist()
    elif mode == "lsp":
        mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
        mat_arr[2] = np.logical_not(mat_arr[2])
        lms = mat_arr.transpose([2, 0, 1])
        kpts = mat_arr.transpose([2, 1, 0]).tolist()

    centers = []
    scales = []
    for idx in range(lms.shape[0]):
        im = Image.open(img_list[idx])
        w = im.size[0]
        h = im.size[1]
        # lsp and lspet dataset doesn't exist groundtruth of center points
        center_x = (lms[idx][0][lms[idx][0] < w].max() +
                    lms[idx][0][lms[idx][0] > 0].min()) / 2
        center_y = (lms[idx][1][lms[idx][1] < h].max() +
                    lms[idx][1][lms[idx][1] > 0].min()) / 2
        centers.append([center_x, center_y])

        scale = (lms[idx][1][lms[idx][1] < h].max() -
                 lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
        scales.append(scale)

    return kpts, centers, scales


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def getBoundingBox(img, kpt, height, width, stride):
    x = []
    y = []

    for index in range(0, len(kpt)):
        if float(kpt[index][1]) >= 0 or float(kpt[index][0]) >= 0:
            x.append(float(kpt[index][1]))
            y.append(float(kpt[index][0]))

    x_min = int(max(min(x), 0))
    x_max = int(min(max(x), width))
    y_min = int(max(min(y), 0))
    y_max = int(min(max(y), height))

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    coord = []
    coord.append([min(int(center_y / stride), height / stride - 1), min(int(center_x / stride), width / stride - 1)])
    coord.append([min(int(y_min / stride), height / stride - 1), min(int(x_min / stride), width / stride - 1)])
    coord.append([min(int(y_min / stride), height / stride - 1), min(int(x_max / stride), width / stride - 1)])
    coord.append([min(int(y_max / stride), height / stride - 1), min(int(x_min / stride), width / stride - 1)])
    coord.append([min(int(y_max / stride), height / stride - 1), min(int(x_max / stride), width / stride - 1)])

    box = np.zeros((int(height / stride), int(width / stride), 5), dtype=np.float32)
    for i in range(5):
        # resize from 368 to 46
        x = int(coord[i][0]) * 1.0
        y = int(coord[i][1]) * 1.0
        heat_map = guassian_kernel(size_h=int(height / stride), size_w=int(width / stride), center_x=x, center_y=y,
                                   sigma=3)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        box[:, :, i] = heat_map

    return box


class LSPDataset:
    """
         0 = Right Ankle
         1 = Right Knee
         2 = Right Hip
         3 = Left  Hip
         4 = Left  Knee
         5 = Left  Ankle
         6 = Right Wrist
         7 = Right Elbow
         8 = Right Shoulder
         9 = Left  Shoulder
        10 = Left  Elbow
        11 = Left  Wrist
        12 = Neck
        13 = Head  Top
    """

    def __init__(self, args, mode, root_dir, sigma, stride, transformer=None):
        assert mode in ["lsp", "lspet"]
        if mode == "lsp":
            root_dir = os.path.join(root_dir, "lsp_dataset")
        else:
            root_dir = os.path.join(root_dir, "lspet_dataset")
        self.height = args.height
        self.width = args.width
        self.img_list = read_data_file(root_dir)
        self.kpt_list, self.center_list, self.scale_list = read_mat_file(mode, root_dir, self.img_list)
        self.stride = stride
        self.transformer = transformer
        self.sigma = sigma
        self.bodyParts = [[13, 12], [12, 9], [12, 8], [8, 7], [9, 10], [7, 6], [10, 11], [12, 3], [2, 3], [2, 1],
                          [1, 0], [3, 4], [4, 5]]

    def __getitem__(self, index):

        img_path = self.img_list[index]
        img = np.array(cv2.resize(cv2.imread(img_path), (self.height, self.width)), dtype=np.float32)
        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]

        # expand dataset
        if self.transformer:
            img, kpt, center = self.transformer(img, kpt, center, scale)
        height, width, _ = img.shape

        heatmap = np.zeros((round(height / self.stride), round(width / self.stride), int(len(kpt) + 1)),
                           dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=int(height / self.stride + 0.5), size_w=int(width / self.stride + 0.5),
                                       center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        img = img.transpose([2, 0, 1])
        img = img / 255
        heatmap = heatmap.transpose([2, 0, 1])

        return img, heatmap

    def __len__(self):
        return len(self.img_list)


def CreateDatasetLSP(args):
    if args.run_modelarts:
        data_root = "/cache/dataset"
        sync_data(args.data_url, data_root)
        args.data_url = os.path.join(data_root, "LSPDataset")

    train_trans = transforms.Compose([
        # transforms.RandomColor(),
        transforms.TypeCast(),
        transforms.RandomRotate(max_degree=10),
        transforms.TestResized(size=(args.height, args.width)),
        transforms.RandomHorizontalFlip()
    ])
    rank_size, rank_id = _get_rank_info()
    train_loader = ds.GeneratorDataset(
        LSPDataset(args, 'lspet', args.data_url, args.sigma, args.stride,
                   transformer=train_trans), column_names=["img", "label"], num_shards=rank_size,
        shard_id=rank_id, shuffle=True, num_parallel_workers=args.num_parallel_workers)

    val_trans = transforms.Compose(
        [transforms.TypeCast(), transforms.TestResized(size=(args.height, args.width))])
    val_loader = ds.GeneratorDataset(
        LSPDataset(args, 'lsp', args.data_url, args.sigma, args.stride, transformer=val_trans),
        column_names=["img", "label"], shuffle=True, num_parallel_workers=args.num_parallel_workers)

    train_dataset = train_loader.batch(batch_size=args.batch_size, drop_remainder=True,
                                       num_parallel_workers=args.num_parallel_workers)
    val_dataset = val_loader.batch(batch_size=args.batch_size, drop_remainder=True,
                                   num_parallel_workers=args.num_parallel_workers)

    return train_dataset, val_dataset


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id


if __name__ == "__main__":
    from src.args import args

    img_path = '../data/LSP/TRAIN/im0001.jpg'
    img = np.array(cv2.resize(cv2.imread(img_path), (368, 368)), dtype=np.float32)

    img = transforms.normalize(transforms.to_tensor(img), [128.0, 128.0, 128.0],
                               [256.0, 256.0, 256.0])
    train_dataset, val_dataset = CreateDatasetLSP(args)
    print(train_dataset.dataset_size)
    print("dataset OK !")

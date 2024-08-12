# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
import os
import random
import gl
import torch.nn.functional as F

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''


def get_bone(jo):
    ntu_pairs = (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
    )
    bone_data_numpy = np.zeros_like(jo)
    for v1, v2 in ntu_pairs:
        bone_data_numpy[:, :, v1 - 1] = jo[:, :, v1 - 1] - jo[:, :, v2 - 1]
    jo = bone_data_numpy
    return jo


def get_vel(ve):
    new_ve = np.zeros_like(ve)
    new_ve[:, :-1] = ve[:, 1:] - ve[:, :-1]
    return new_ve


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch


class NTU_RGBD_Dataset(data.Dataset):

    def __init__(self, mode='train', data_list=None, debug=False, extract_frame=1, modal=1, bone=0, vel=0, process=0,
                 weighted=0):
        global path
        self.modal = modal
        self.bone = bone
        self.vel = vel
        self.process = process
        self.weighted = weighted
        super(NTU_RGBD_Dataset, self).__init__()
        ori = "Path of the dataset"

        if gl.dataset == 'ntu-T':
            path = "ntu120/NTU-T"
            segment = 30
        elif gl.dataset == 'ntu-S':
            path = "ntu120/NTU-S"
            segment = 30
        elif gl.dataset == 'kinetics':
            path = "kinetics/Kinetics"
            segment = 50
        elif gl.dataset == 'pku':
            path = "mypku"
            segment = 30
        elif gl.dataset == 'ntu1s':
            path = "ntu1s"
            segment = 30
            extract_frame = 3
        else:
            ValueError('Unknown dataset')

        if mode == 'train':
            data_path = os.path.join(ori, path, 'train_data.npy')
            label_path = os.path.join(ori, path, 'train_label.npy')
            num_frame = os.path.join(ori, path, 'train_frame.npy')
        elif mode == 'val':
            data_path = os.path.join(ori, path, 'val_data.npy')
            label_path = os.path.join(ori, path, 'val_label.npy')
            num_frame = os.path.join(ori, path, 'val_frame.npy')
        elif gl.dataset == 'ntu1s' and mode == 'test':
            data_path = os.path.join(ori, path, 'val_data.npy')
            label_path = os.path.join(ori, path, 'val_label.npy')
            num_frame = os.path.join(ori, path, 'val_frame.npy')
        else:
            data_path = os.path.join(ori, path, 'test_data.npy')
            label_path = os.path.join(ori, path, 'test_label.npy')
            num_frame = os.path.join(ori, path, 'test_frame.npy')

        self.data, self.label, self.num_frame = np.load(data_path), np.load(label_path), np.load(num_frame)

        if debug:  # debug模式使用少量数据
            data_len = len(self.label)
            data_len = int(0.1 * data_len)
            self.label = self.label[0:data_len]
            self.data = self.data[0:data_len]
            self.num_frame = self.num_frame[0:data_len]

        if extract_frame == 1:  # 从序列抽取帧
            self.data = self.extract_frame(self.data, self.num_frame, segment)
        elif extract_frame == 3:
            self.data = self.extract_frame2(self.data, self.num_frame, segment)
        print('sample_num in {}'.format(mode), len(self.label))
        n_classes = len(np.unique(self.label))
        print('n_class', n_classes)

    def __getitem__(self, idx):  # 获取样本和标签
        x = self.data[idx]  # x.shape:(3, 30, 25, 2)
        if gl.dataset == 'ntu1s':
            x = random_rot(x)
        if self.process == 1:
            if self.modal == 2:
                x1 = get_bone(x)
                return x, x1, self.label[idx]
            elif self.modal == 3:
                x1 = get_bone(x)
                x2 = get_vel(x)
                return x, x1, x2, self.label[idx]
            elif self.modal == 4:
                x1 = get_bone(x)
                x2 = get_vel(x)
                x3 = get_vel(x1)
                return x, x1, x2, x3, self.label[idx]
            else:
                ValueError('Unknown modal, you should choose 2, 3 or 4.')
        if self.weighted == 1:
            if self.modal == 2:
                x1 = get_bone(x)
                x = x + x1
                return x, self.label[idx]
            elif self.modal == 3:
                x1 = get_bone(x)
                x2 = get_vel(x)
                x = x * 0.6 + x1 * 0.4 + x2 * 0.4
                return x, self.label[idx]
            elif self.modal == 4:
                x1 = get_bone(x)
                x2 = get_vel(x)
                x3 = get_vel(x1)
                x = x * 0.5 + x1 * 0.3 + x2 * 0.3 + x3 * 0.3
                return x, self.label[idx]
            else:
                ValueError('You have set the weighted parameter and want to use the early fusion method of weighted '
                           'addition, but the number of modalities you have selected is not 2, 3, 4. '
                           'Please set the modal parameter')
        if self.modal == 2:
            x1 = get_bone(x)
            x = np.concatenate((x, x1), axis=0)
            return x, self.label[idx]
        elif self.modal == 3:
            x1 = get_bone(x)
            x2 = get_vel(x)
            x = np.concatenate((x, x1, x2), axis=0)
            return x, self.label[idx]
        elif self.modal == 4:
            x1 = get_bone(x)
            x2 = get_vel(x)
            x3 = get_vel(x1)
            x = np.concatenate((x, x1, x2, x3), axis=0)
            return x, self.label[idx]
        if self.bone == 1:
            x = get_bone(x)
        if self.vel == 1:
            x = get_vel(x)

        return x, self.label[idx]

    def __len__(self):  # 数据集长度
        return len(self.label)

    def extract_frame(self, x, num_frame, segment):  # 从序列抽取帧
        n, c, t, v, m = x.shape
        #(2320, 3, 300, 25, 2)
        # 验证样本数与帧数数组的长度一致
        assert n == len(num_frame)

        num_frame = np.array(num_frame)
        # 计算每段抽几帧,四舍五入取整
        step = num_frame // segment
        new_x = []
        # 遍历每个样本
        for i in range(n):
            # 如果总帧数少于要抽取的帧数
            if num_frame[i] < segment:
                # 直接取完整帧addData
                new_x.append(np.expand_dims(x[i, :, 0:segment, :, :], 0).reshape(1, c, segment, v, m))
                # 跳过当前样本
                continue
            # 随机采样抽取相应段数的帧
            idx = [random.randint(j * step[i], (j + 1) * step[i] - 1) for j in range(segment)]
            # 添加抽取的帧数据
            new_x.append(np.expand_dims(x[i, :, idx, :, :], 0).reshape(1, c, segment, v, m))
        # 拼接所有样本的抽取帧
        new_x = np.concatenate(new_x, 0)
        return new_x

    def extract_frame2(self, x, num_frame, segment):  # 从序列抽取帧
        n, c, t, v, m = x.shape
        #(2320, 3, 300, 25, 2)
        # 验证样本数与帧数数组的长度一致
        assert n == len(num_frame)

        num_frame = np.array(num_frame)
        # 计算每段抽几帧,四舍五入取整
        step = num_frame // segment
        new_x = []
        # 遍历每个样本
        for i in range(n):
            # 如果总帧数少于要抽取的帧数
            if num_frame[i] < segment:
                # Interpolate to match the segment length
                data = torch.tensor(x[i, :, :num_frame[i], :, :], dtype=torch.float)
                data = data.permute(0, 2, 3, 1).contiguous().view(c * v * m, num_frame[i])
                data = data[None, None, :, :]
                data = F.interpolate(data, size=(c * v * m, segment), mode='bilinear',
                                     align_corners=False).squeeze()
                data = data.contiguous().view(c, v, m, segment).permute(0, 3, 1, 2).contiguous().numpy()
                new_x.append(np.expand_dims(data, 0).reshape(1, c, segment, v, m))
                continue
            # 随机采样抽取相应段数的帧
            idx = [random.randint(j * step[i], (j + 1) * step[i] - 1) for j in range(segment)]
            # 添加抽取的帧数据
            new_x.append(np.expand_dims(x[i, :, idx, :, :], 0).reshape(1, c, segment, v, m))
        # 拼接所有样本的抽取帧
        new_x = np.concatenate(new_x, 0)
        return new_x

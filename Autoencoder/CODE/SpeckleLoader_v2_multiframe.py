import os
import random

import numpy as np
import scipy.io as io
from imageio import imread

import torch
from torch.utils import data

from data.preprocess import TRAIN_AUGS_2D
from data.preprocess import TEST_AUGS_2D
from data.preprocess import mat2npy


def find_classes(path):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset_first_imgs(path):
    i = 0
    j = 0
    imgs = []
    lutb1 = {}
    lutb2 = {}
    first_imgs = []
    path = os.path.expanduser(path)

    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                i += 1
                mat_path = os.path.join(root, fname)
                imgs.append(mat_path)
                if len(imgs) == 1:
                    elements = imgs[0].split('/')
                    lutb2[elements[-2]] = i
                    first_imgs += imgs
                    imgs = []

                    if i % 4 == 0:
                        lutb1[elements[-3]] = j * 4
                        i = 0; j += 1
                    break;

    return first_imgs, lutb1, lutb2


def make_dataset_met1(path, class_to_idx, img_T, frames, valid='34', test=False, flow=False):
    imgs = []
    clips = []

    clips_train = []
    clips_valid = []

    valid = valid.split(',')
    valid_1, valid_2 = valid[0], valid[1]
    valid = [valid_1, valid_2]

    path = os.path.expanduser(path)
    if flow:
        num = 499 // img_T
    else:
        num = 500 // img_T * img_T

    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                imgs.append(item)

                if len(imgs) != 0 and len(imgs) % frames == 0:
                    sample_label = root.split('/')[-2]
                    if not test:
                        if sample_label in valid:
                            clips_valid.append(imgs)
                            imgs = []
                            if len(clips_valid) % num == 0: break;

                        else:
                            clips_train.append(imgs)
                            imgs = []
                            if len(clips_train) % num == 0: break;

                    else:
                        if sample_label in valid:
                            clips_valid.append(imgs)
                            imgs = []
                            if len(clips_valid) % num == 0: break;

                        else:
                            imgs = []

    return clips_train, clips_valid


def SpeckleDataset(root, T, frames, valid, fold, method, test=False, flow=False):
    classes, class_to_idx = find_classes(root)
    first_imgs, lutb1, lutb2 = make_dataset_first_imgs(root)

    if method==0:
        tr_imgs, val_imgs = make_dataset_met1(root, class_to_idx, T, frames, valid, test=test, flow=flow)

    elif method==1:
        tr_imgs, val_imgs = make_dataset_met1(root, class_to_idx, T, frames, valid, test=test, flow=flow)
#        val_apr_root='/data/0419_speckles/106/'
        val_may_root='/data/0502_speckles/106/'
        val_imgs, _ = make_dataset_met1(val_may_root, class_to_idx, T, frames, valid, test=False, flow=flow)

    elif method==2:
        root2='/data/0419_speckles/106/'
        tr_imgs, val_imgs = make_dataset_met1(root, class_to_idx, T, frames, valid, test=test, flow=flow)
        tr_imgs2, val_imgs2 = make_dataset_met1(root2, class_to_idx, T, frames, valid, test=test, flow=flow)
        tr_imgs += tr_imgs2
        val_imgs += val_imgs2

#        val_may_root='/data/0530_speckles/106/'
#        val_imgs, _ = make_dataset_met1(val_may_root, class_to_idx, T, frames, valid, test=False, flow=flow)

    elif method==3:
        root2='/data/0502_speckles/106/'
        root3='/data/0530_speckles/106/'
        tr_imgs, val_imgs = make_dataset_met1(root, class_to_idx, T, frames, valid, test=test, flow=flow)
        tr_imgs2, val_imgs2 = make_dataset_met1(val_may_root, class_to_idx, T, frames, valid, test=test, flow=flow)
        tr_imgs3, val_imgs3 = make_dataset_met1(val_may_root, class_to_idx, T, frames, valid, test=test, flow=flow)
        tr_imgs += tr_imgs2 + tr_imgs3
        val_imgs += val_imgs2 + val_imgs3

    origin_imgs = len(tr_imgs) + len(val_imgs)
    print("{} origin : {}, aug: (Tr {}, Val {})".format(
            root, origin_imgs, len(tr_imgs), len(val_imgs))
    )

    return tr_imgs, val_imgs, first_imgs, lutb1, lutb2


class _Dataset(data.Dataset):
    def __init__(self, img, first_imgs, lutb1, lutb2, frames, aug_rate=1.0, transform=None):
        self.imgs = img
        self.f_imgs = []
        self.frames = frames
        self.origin_imgs = len(img)
        self.augs = [] if transform is None else transform

        self.lutb1 = lutb1
        self.lutb2 = lutb2

#        if aug_rate != 0:
#            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

        for img in first_imgs:
            self.f_imgs.append(np.array(imread(img)))
        self.name_f_imgs = first_imgs
        self.f_imgs = np.stack(self.f_imgs, axis=0)

    def __getitem__(self, index):
        num_imgs = len(self.imgs[index])

        # Extract Sequential T Images
        img_infos = np.array(self.imgs[index])
        paths = img_infos[:, 0].tolist()
        target = img_infos[:, 1][0]

#        elements = path.split('/')
#        no_sample = elements[-3]
#        cam_label = elements[-2]
#        idx_f_img = self.lutb1[no_sample] + self.lutb2[cam_label] - 1

        imgs = []
        f_img = np.array(imread(paths[0]))
#        imgs.append(f_img)
#        paths = paths[0:]
        for path in paths:
            img = np.array(imread(path))
#            img -= f_img
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)

        if index > self.origin_imgs:
            for t in self.augs:
                imgs = t(imgs)

        else:
            for t in TEST_AUGS_2D:
                imgs = t(imgs)

        return imgs, target, paths

    def __len__(self):
        return len(self.imgs)


class _Dataset_optical_flow(data.Dataset):
    def __init__(self, imgs, frames, aug_rate=0, transform=None):
        self.imgs = imgs
        self.frames = frames
        self.origin_imgs = len(self.imgs)
        self.augs = [] if transform is None else transform

        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

    def __getitem__(self, index):
        # Extract Sequential T Images
        imgs = np.array(self.imgs[index])
        path = imgs[:, 0].tolist()
        target = imgs[:, 1][0]

        img = []

        for i in range(len(path)):
            f = open(path[i])
            magic = np.fromfile(f, np.float32, count=1)
            data2d = None
            if 202021.25 != magic:
                print ('Magic number incorrect. Invalid .flo file')
            else:
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                data2d = np.fromfile(f, np.float32, count=2 * w * h)
                # reshape data into 3D array (columns, rows, channels)
                data2d = np.resize(data2d, (h, w, 2))
                image = data2d
            img.append(image)
        img = np.array(img)

        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img, flow=True)
        else:
            for t in TEST_AUGS_2D:
                img = t(img, flow=True)

        return img, target, path

    def __len__(self):
        return len(self.imgs)


def _make_weighted_sampler(images, classes):
    nclasses = len(classes)
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    print(classes)
    print(count)

    N = float(sum(count))
    assert N == len(images)

    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    return sampler


def SpeckleLoader(image_path, frames, batch_size, test_batch_size, T, sampler=False,
                 tr_transform=None, val_transform=None, test_transform=None,
                 tr_aug_rate=0, val_aug_rate=0, test_aug_rate=0, fold=None, flow=False,
                 num_workers=1, shuffle=False, drop_last=False, test=False, val_sample=None, method=1):

    tr_imgs, val_imgs, f_imgs, lutb1, lutb2 = SpeckleDataset(image_path, T, frames, val_sample, fold, method, test=test, flow=flow)

    if not test:
        if flow:
            tr_dataset = _Dataset_optical_flow(tr_imgs, frames, transform=tr_transform)
            val_dataset = _Dataset_optical_flow(val_imgs, frames, aug_rate=0., transform=val_transform)

        else:
            tr_dataset = _Dataset(tr_imgs, f_imgs, lutb1, lutb2, frames, transform=tr_transform)
            val_dataset = _Dataset(val_imgs, f_imgs, lutb1, lutb2, frames, aug_rate=0., transform=val_transform)

    else:
        if flow:
            val_dataset = _Dataset_optical_flow(val_imgs, frames, aug_rate=0., transform=tr_transform)
        else:
            val_dataset = _Dataset(val_imgs, f_imgs, lutb1, lutb2, frames, aug_rate=0., transform=tr_transform)

    print('Data len: {} Train, {} Valid'.format(len(tr_imgs), len(val_imgs)))

    if sampler:
        print("Sampler : ", image_path[-5:])
        tr_sampler = _make_weighted_sampler(tr_dataset.imgs)
        val_sampler = _make_weighted_sampler(val_dataset.imgs)
        tr_dataset = data.DataLoader(tr_dataset, batch_size, sampler=sampler, num_workers=num_workers)
        val_dataset = data.DataLoader(val_dataset, batch_size, sampler=sampler, num_workers=num_workers)
        return tr_dataset, val_dataset


    if not test:
        tr_dataset = data.DataLoader(tr_dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
        val_dataset = data.DataLoader(val_dataset, test_batch_size, num_workers=num_workers)
        return tr_dataset, val_dataset

    else:
        val_dataset = data.DataLoader(val_dataset, test_batch_size, num_workers=num_workers)
        return val_dataset

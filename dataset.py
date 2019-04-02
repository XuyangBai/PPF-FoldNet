import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 num_points=2500,
                 classification=False,
                 class_choice=None,
                 data_augmentation=True):
        self.num_points = num_points
        self.root = root
        self.split = split
        self.cat2id = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        # parse category file.
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = ls[1]

        # parse segment num file.
        with open('misc/num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])

        # if a subset of classes is specified.
        if class_choice is not None:
            self.cat2id = {k: v for k, v in self.cat2id.items() if k in class_choice}
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        self.datapath = []
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        filelist = json.load(open(splitfile, 'r'))
        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat2id.values():
                self.datapath.append([
                    self.id2cat[category],
                    os.path.join(self.root, category, 'points', uuid + '.pts'),
                    os.path.join(self.root, category, 'points_label', uuid + '.seg')
                ])

        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))
        # print("classes:", self.classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)

        # randomly sample self.num_points point from the origin point cloud.
        choice = np.random.choice(len(seg), self.num_points, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation and self.split == 'train':
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # TODO: why only rotate the x and z axis??
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


# class 3DMatchDataset(data.Dataset):


if __name__ == '__main__':
    dataset = "shapenet"
    datapath = "./data/shapenetcore_partanno_segmentation_benchmark_v0"

    if dataset == 'shapenet':
        print("Segmentation task:")
        d = ShapeNetDataset(root=datapath, num_points=2048, class_choice=['Chair'], classification=False)
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(), seg.type())

        print("Classification task:")
        d = ShapeNetDataset(root=datapath, num_points=2048, classification=True)
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(), cls.type())

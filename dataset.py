import torch.utils.data as data
import os
import os.path
import open3d
import numpy as np
import sys
from tqdm import tqdm
import json
from input_preparation import input_preprocess


class SunDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 num_patches=32,  # num of patches per point cloud.
                 data_augmentation=True):
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.num_patches = num_patches

        self.ids_list = [filename.split(".")[0] for filename in os.listdir(self.root)]
        self.ids_list = sorted(list(set(self.ids_list)))
        if split == 'train':
            self.ids_list = self.ids_list[0: int(0.8 * len(self.ids_list))]
        else:
            self.ids_list = self.ids_list[int(0.8 * len(self.ids_list)):-1]

    def __getitem__(self, index):
        ind = np.random.choice(range(2048), self.num_patches, replace=False)
        patches = np.load(os.path.join(self.root, self.ids_list[index] + ".npy"))
        return patches[ind], self.ids_list[index]
        # if self.split == 'train':
        #     patches = np.load(os.path.join(self.root, self.ids_list[index] + ".npy"))
        #     return patches
        # else:
        #     patches = np.load(os.path.join(self.root, self.ids_list[index] + ".npy"))
        #     pcd = open3d.read_point_cloud(os.path.join(self.root, self.ids_list[index] + ".pcd"))
        #     return patches, [pcd]

    def __len__(self):
        return len(self.ids_list)


if __name__ == '__main__':
    datapath = "./data/train/sun3d-harvard_c11-hv_c11_2/seq-01-test-processed"
    d = SunDataset(root=datapath, split='train')
    patches, id = d[0]
    assert patches.shape == (32, 1024, 4)
    print(id)
    print(d.ids_list)

    # d = SunDataset(root=datapath, split='test')
    # patches, pcd = d[0]
    # assert patches.shape == (2048, 1024, 4)
    # print(pcd)

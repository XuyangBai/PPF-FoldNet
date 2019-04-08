import torch.utils.data as data
import os
import os.path
import torch
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

        self.ids_list = [filename for filename in os.listdir(self.root)]

    def __getitem__(self, index):
        patches = np.load(os.path.join(self.root, self.ids_list[index]))
        return patches[np.random.choice(range(2048), self.num_patches, replace=False)]

    def __len__(self):
        return len(self.ids_list)


if __name__ == '__main__':
    dataset = "sun3d"
    if dataset == 'sun3d':
        datapath = "./data/sun3d-harvard_c11-hv_c11_2/seq-01-test-npy"
        d = SunDataset(root=datapath, num_patches=32)
        patches = d[0]
        assert patches.shape == (32, 1024, 4)
    else:
        print("No such dataset")

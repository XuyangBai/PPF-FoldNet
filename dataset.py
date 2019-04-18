import torch.utils.data as data
import os
import os.path
import open3d
import numpy as np
import time
from tqdm import tqdm
import json
from input_preparation import get_local_patches_on_the_fly


class SunDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 num_patches=32,  # num of patches per point cloud. which is also the batch size of the input.
                 num_points_per_patch=1024,
                 data_augmentation=True,
                 on_the_fly=True):
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.num_patches = num_patches
        self.num_points_per_patch = num_points_per_patch
        self.on_the_fly = on_the_fly

        # Support the whole 3Dmatch dataset
        with open(os.path.join(root, f'scene_list_{split}.txt')) as f:
            scene_list = f.readlines()

        self.ids_list = []
        for scene in scene_list:
            ids = [scene + "/seq-01/" + str(filename.split(".")[0]) for filename in os.listdir(os.path.join(self.root, scene + '/seq-01/'))]
            self.ids_list += sorted(list(set(ids)))

        # self.ids_list = [filename.split(".")[0] for filename in os.listdir(self.root)]
        # self.ids_list = sorted(list(set(self.ids_list)))
        # if split == 'train':
        #     self.ids_list = self.ids_list[0: int(0.8 * len(self.ids_list))]
        # else:
        #     self.ids_list = self.ids_list[int(0.8 * len(self.ids_list)):-1]

    def __getitem__(self, index):
        id = self.ids_list[index]
        if self.on_the_fly:
            return get_local_patches_on_the_fly(self.root, id, self.num_patches, self.num_points_per_patch), id

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
    # datapath = "./data/train/sun3d-harvard_c11-hv_c11_2/seq-01-train"
    # d = SunDataset(root=datapath, split='train', on_the_fly=True)
    # print(d.ids_list)
    # start_time = time.time()
    # for i in range(10):
    #     patches, id = d[0]
    # print(f"On the fly: {time.time() - start_time}")
    #
    # datapath = "./data/train/sun3d-harvard_c11-hv_c11_2/seq-01-train-processed"
    # d = SunDataset(root=datapath, split='train', on_the_fly=False)
    # print(d.ids_list)
    # start_time = time.time()
    # for i in range(10):
    #     patches, id = d[0]
    # print(f"Not On the fly: {time.time() - start_time}")

    datapath = "./data/train/"
    d = SunDataset(root=datapath, split='train', on_the_fly=True)
    print(d.ids_list)
    patches, id = d[0]
    print(patches.shape)
    print(id)

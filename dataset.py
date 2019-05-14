import torch.utils.data as data
import os
import os.path
import open3d
import numpy as np
import time
from tqdm import tqdm
import json
import numpy
from input_preparation import *


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
        self.scene_list = []
        for scene in scene_list:
            if not scene.__contains__('sun3d'):
                continue
            scene = scene.replace("\n", "")
            ids = [scene + "/seq-01/" + str(filename.split(".")[0]) for filename in os.listdir(os.path.join(self.root, scene + '/seq-01/'))]
            self.ids_list += sorted(list(set(ids)))
            self.scene_list.append(scene)
        # if split == 'test':
        #    self.ids_list = self.ids_list[0:10000]
        # if split == 'train':
        #    self.ids_list = self.ids_list[0:50000]

    def __getitem__(self, index):
        id = self.ids_list[index]
        if self.on_the_fly:
            try:
                return get_local_patches_on_the_fly(self.root, id, self.num_patches, self.num_points_per_patch), id
            except:
                print(id, "cannot open")
                return self.__getitem__(0)

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


class SunDataset_Supervised(data.Dataset):
    def __init__(self, root, split, num_patches=32, num_points_per_patch=1024):
        self.root = root
        self.split = split
        self.num_patches = num_patches
        self.num_points_per_patch = num_points_per_patch

        with open(os.path.join(root, f'scene_list_{split}.txt')) as f:
            scene_list = f.readlines()

        self.ids_list = []
        self.scene_list = []
        self.scene_to_ids = {}
        for scene in scene_list:
            if not scene.__contains__('sun3d'):
                continue
            scene = scene.replace("\n", "")
            ids = [scene + "/seq-01/" + str(filename.split(".")[0]) for filename in os.listdir(os.path.join(self.root, scene + '/seq-01/'))]
            self.ids_list += sorted(list(set(ids)))
            self.scene_list.append(scene)
            self.scene_to_ids[scene] = sorted(list(set(ids)))

            if split == 'train' and len(self.ids_list) > 50000:
                break
            if split == 'test' and len(self.ids_list) > 1000:
                break

    def __getitem__(self, index):
        try:
            anc_id = self.ids_list[index]
            anchor_scene = anc_id.split("/")[0]
            # training set select positive point randomly
            if self.split == 'train':
                pos_id = np.random.choice(self.scene_to_ids[anchor_scene])
                random_state = None
            # for test set we should select positive point fixed. So create a random state
            else:
                random_state = np.random.RandomState(51)
                pos_id = random_state.choice(self.scene_to_ids[anchor_scene])
            anchor, positive = get_positive(self.root, anc_id, pos_id, self.num_patches, random_state)
            return anchor, positive
        except:
            print(f"get item {self.ids_list[index]} error")
            return self.__getitem__(np.random.choice(len(self.ids_list)))

    def __len__(self):
        return len(self.ids_list)


def get_positive(root, anc_id, pos_id, num_patches, random_state=None):
    # build point cloud from rgbd file
    anc_pcd = rgbd_to_point_cloud(root, anc_id)
    pos_pcd = rgbd_to_point_cloud(root, pos_id)
    cal_local_normal(anc_pcd)
    cal_local_normal(pos_pcd)

    # select [num_patches] points from anchor point cloud
    if random_state is not None:
        anc_ind = random_state.choice(range(len(anc_pcd.points)), num_patches, replace=False)
    else:
        anc_ind = np.random.choice(range(len(anc_pcd.points)), num_patches, replace=False)
        random_state = np.random.RandomState(51)

    anc_pts = open3d.geometry.select_down_sample(anc_pcd, anc_ind)

    # build anchor local patches
    anchor_neighbor = collect_local_neighbor(anc_pts, anc_pcd, random_state=random_state)
    anchor_local_patch = build_local_patch(anc_pts, anc_pcd, anchor_neighbor)

    # for each points in anchor, find the nn in positive point cloud
    # TODO: there might be some repitition.
    kdtree = open3d.geometry.KDTreeFlann(pos_pcd)
    pos_ind = []
    for pts in anc_pts.points:
        _, ind, dis = kdtree.search_knn_vector_3d(pts, 1)
        pos_ind.append(ind[0])
        # _, ind, dis = kdtree.search_knn_vector_3d(pts, 2)
        # if dis[0] < 0.7 * dis[1]:
        #     pos_ind.append(ind[0])

    # build the positive local patches
    pos_pts = open3d.geometry.select_down_sample(pos_pcd, pos_ind)
    positive_neighbor = collect_local_neighbor(pos_pts, pos_pcd, random_state=random_state)
    positive_local_patch = build_local_patch(pos_pts, pos_pcd, positive_neighbor)

    return anchor_local_patch, positive_local_patch


if __name__ == '__main__':
    # test SunDataset
    # datapath = "/data/3DMatch/whole"
    # d = SunDataset(root=datapath, split='test', on_the_fly=True)
    # print(len(d.ids_list))
    # # print(d.scene_list)
    # start_time = time.time()
    # for i in range(len(d.ids_list)):
    #     patches, id = d[i]
    #     if i % 100 == 0:
    #         print(f"{i} : {time.time() - start_time} s")
    # print(f"Test set On the fly: {time.time() - start_time}")
    #
    # datapath = "/data/3DMatch/whole"
    # d = SunDataset(root=datapath, split='train', on_the_fly=True)
    # print(len(d.ids_list))
    # # print(d.scene_list)
    # start_time = time.time()
    # for i in range(len(d.ids_list)):
    #     patches, id = d[i]
    #     if i % 100 == 0:
    #         print(f"{i}: {time.time() - start_time} s")
    # print(f"Training set On the fly: {time.time() - start_time}")

    # test SunDataset_Supervised time
    start_time = time.time()
    datapath = './data/3DMatch'
    d = SunDataset_Supervised(root=datapath, split='train')
    for i in range(len(d.ids_list)):
        patches, id = d[i]
        if i == 10:
            break
    print(time.time() - start_time)

    # test SunDataset_Supervised testset fixed or not
    d = SunDataset_Supervised(root=datapath, split='test')
    anc, pos = d[0]
    new_d = SunDataset_Supervised(root=datapath, split='test')
    anc_new, pos_new = d[0]
    assert np.array_equal(anc, anc_new)
    assert np.array_equal(pos, pos_new)

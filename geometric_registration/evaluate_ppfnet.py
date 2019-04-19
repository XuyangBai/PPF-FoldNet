import open3d
import numpy as np
import time
import os
from dataloader import get_dataloader
from geometric_registration.utils import get_pcd, get_keypts, get_desc
from input_preparation import rgbd_to_point_cloud
from scipy.spatial import KDTree


def calculate_M(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 512]
    """
    # start_time = time.time()
    # result = []
    # for fea1, i in zip(source_desc, range(len(source_desc))):
    #     kdtree_s = KDTree(target_desc)
    #     dis, idx1 = kdtree_s.query(fea1, 1)
    #     fea2 = target_desc[idx1]
    #     kdtree_t = KDTree(source_desc)
    #     dis, idx2 = kdtree_t.query(fea2, 1)
    #     if i == idx2:
    #         result.append([i, idx1])
    # print(time.time() - start_time)
    # return np.array(result)

    kdtree_s = KDTree(target_desc)
    sourceNNdis, sourceNNidx = kdtree_s.query(source_desc, 1)
    kdtree_t = KDTree(source_desc)
    targetNNdis, targetNNidx = kdtree_t.query(target_desc, 1)
    result = []
    for i in range(len(sourceNNidx)):
        if targetNNidx[sourceNNidx[i]] == i:
            result.append([i, sourceNNidx[i]])
    return np.array(result)


def calculate_M_gnd(correspondence, source, target, tao1):
    def distance(p1, p2):
        return np.sqrt(np.dot(p1 - p2, p1 - p2))

    result = []
    for pair in correspondence:
        if distance(source[pair[0]], target[pair[1]]) <= tao1:
            result.append([pair[0], pair[1]])
    return np.array(result)


def is_matching_pairs(source, target, threshold=0.02):
    trans_init = np.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    evaluation = open3d.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation.fitness)
    if evaluation.fitness >= 0.3:
        return True
    else:
        return False


def evaluate(model, loader, tao1=0.3, tao2=0.05):
    # model.eval()
    total_matching_fragments = 0
    pred_matching_fragments = 0
    for iter, inputs in enumerate(loader):
        # assume P and Q have already been transformed to the canonical coordinate.
        P = inputs[0]
        Q = inputs[1]
        if not is_matching_pairs(P, Q):
            continue
        else:
            total_matching_fragments += 1
        g_P = model.encoder(P)
        g_Q = model.encoder(Q)
        g_P = P
        g_Q = Q
        matching_points = calculate_M(g_P, g_Q)
        gt_matching_points = calculate_M_gnd(matching_points, P, Q, tao1)
        if len(gt_matching_points) * 1.0 / len(matching_points) > tao2:
            pred_matching_fragments += 1
    recall = pred_matching_fragments * 1.0 / total_matching_fragments
    print(f"Recall with tao1 = {tao1*100}cm and tao2 = {tao2} is: {recall}")


if __name__ == '__main__':
    scene_list = [
        # '7-scenes-redkitchen',
        # 'sun3d-home_at-home_at_scan1_2013_jan_1',
        # 'sun3d-home_md-home_md_scan9_2012_sep_30',
        # 'sun3d-hotel_uc-scan3',
        # 'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        # 'sun3d-mit_76_studyroom-76-1studyroom2',
        # 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    # datapath = "./data/test/sun3d-hotel_umd-maryland_hotel3/"
    # interpath = "./data/intermediate-files-real/sun3d-hotel_umd-maryland_hotel3/"
    # savepath = "./data/intermediate-files-real/sun3d-hotel_umd-maryland_hotel3/"
    for scene in scene_list:
        pcdpath = f"./fragments/{scene}/"
        interpath = f"./intermediate-files-real/{scene}/"
        keyptspath = os.path.join(interpath, "keypoints/")
        descpath = os.path.join(interpath, "ppf_desc/")

        cloud_bin_s = f'cloud_bin_1'
        cloud_bin_t = f'cloud_bin_2'
        # 1. load point cloud, keypoints, descriptors
        original_source_pc = get_pcd(pcdpath, cloud_bin_s)
        original_target_pc = get_pcd(pcdpath, cloud_bin_t)
        print("original source:", original_source_pc)
        print("original target:", original_target_pc)
        # downsample and estimate the normals
        voxel_size = 0.02
        source_pc = open3d.geometry.voxel_down_sample(original_source_pc, voxel_size)
        open3d.geometry.estimate_normals(source_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target_pc = open3d.geometry.voxel_down_sample(original_target_pc, voxel_size)
        open3d.geometry.estimate_normals(target_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        print("downsampled source:", source_pc)
        print("downsampled target:", target_pc)
        # load keypoints and descriptors
        source_keypts = get_keypts(keyptspath, cloud_bin_s)
        target_keypts = get_keypts(keyptspath, cloud_bin_t)
        # print(source_keypts.shape)
        source_desc = get_desc(descpath, cloud_bin_s, desc_name='ppf')
        target_desc = get_desc(descpath, cloud_bin_t, desc_name='ppf')
        # trans = ransac_based_on_correspondence(source_keypts[0:1000], target_keypts[0:1000], source_desc[0:1000], target_desc[0:1000])

        corr = calculate_M(source_desc[0:500], target_desc[0:500])
        # M = len(corr)
        # # TODO: 需要先用ground truth的transformation对source进行变换
        # corr_gnd = calculate_M_gnd(corr, source_keypts[0:500], target_keypts[0:500], 0.3)
        # M_gnd = len(corr_gnd)

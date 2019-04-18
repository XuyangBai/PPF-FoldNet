import open3d
import os
import time
import numpy as np
from scipy.spatial import KDTree

from geometric_registration.utils import get_pcd, get_keypts, get_desc

np.random.seed(0)


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

    start_time = time.time()
    kdtree_s = KDTree(target_desc)
    sourceNNdis, sourceNNidx = kdtree_s.query(source_desc, 1)
    kdtree_t = KDTree(source_desc)
    targetNNdis, targetNNidx = kdtree_t.query(target_desc, 1)
    result = []
    # print(time.time() - start_time)
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


def ransac_based_on_feature_matching(source_keypts, target_keypts, source_desc, target_desc):
    source_pc = open3d.PointCloud()
    source_pc.points = open3d.utility.Vector3dVector(source_keypts)
    target_pc = open3d.PointCloud()
    target_pc.points = open3d.utility.Vector3dVector(target_keypts)
    source_feature = open3d.registration.Feature()
    source_feature.data = source_desc.transpose()
    target_feature = open3d.registration.Feature()
    target_feature.data = target_desc.transpose()

    voxel_size = 0.05
    # source_pc = open3d.geometry.voxel_down_sample(source_pc, voxel_size)
    open3d.geometry.estimate_normals(source_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    # target_pc = open3d.geometry.voxel_down_sample(target_pc, voxel_size)
    open3d.geometry.estimate_normals(target_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    # source_feature = open3d.registration.compute_fpfh_feature(source_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    # target_feature = open3d.registration.compute_fpfh_feature(target_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    result = open3d.registration_ransac_based_on_feature_matching(
        source=source_pc,
        target=target_pc,
        source_feature=source_feature,
        target_feature=target_feature,
        max_correspondence_distance=0.05,
        estimation_method=open3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[],
        criteria=open3d.registration.RANSACConvergenceCriteria(4000000, 500)
        # source_feature=source_feature,
        # target_feature=target_feature,
        # max_correspondence_distance=max_dis,
        # ransac_n=ransac_n,
        # criteria=open3d.registration.RANSACConvergenceCriteria(10000, 500)
    )
    return result


def ransac_based_on_correspondence(source_keypts, target_keypts, source_desc, target_desc):
    source_pc = open3d.PointCloud()
    source_pc.points = open3d.utility.Vector3dVector(source_keypts)
    target_pc = open3d.PointCloud()
    target_pc.points = open3d.utility.Vector3dVector(target_keypts)

    distance_threshold = 0.05
    voxel_size = 0.05

    # source_pc = open3d.geometry.voxel_down_sample(source_pc, voxel_size)
    open3d.geometry.estimate_normals(source_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    # target_pc = open3d.geometry.voxel_down_sample(target_pc, voxel_size)
    open3d.geometry.estimate_normals(target_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    correspondence = calculate_M(source_desc, target_desc)
    print("Num of correspondence", len(correspondence))

    result = open3d.registration_ransac_based_on_correspondence(
        source=source_pc,
        target=target_pc,
        corres=open3d.utility.Vector2iVector(correspondence),
        max_correspondence_distance=distance_threshold,
        estimation_method=open3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=6,
        criteria=open3d.registration.RANSACConvergenceCriteria(10000, 500)
    )
    print(result)
    print(result.correspondence_set)
    return result.transformation


def calculate_overlap(pc1, pc2, trans):
    pc1.transform(trans)
    kdtree = KDTree(pc2)
    dis = kdtree.query(pc1, 1)
    count = np.sum(dis > 0.05)

    pass


def icp_refine(source_pc, target_pc, trans, dis):
    """
    use icp to refine the rigid transformation.
    """
    result = open3d.registration_icp(
        source=source_pc,
        target=target_pc,
        max_correspondence_distance=dis,
        init=trans,
        estimation_method=open3d.TransformationEstimationPointToPoint(False)
    )
    return result


def cal_alignment(source_pc, target_pc, distance=0.05):
    source_pts = np.asarray(source_pc.points)
    target_pts = np.asarray(target_pc.points)
    kdtree_s = KDTree(target_pts)

    sourceNNdis, sourceNNidx = kdtree_s.query(source_pts, 1)
    ratio = float(np.sum(sourceNNdis < distance) / len(source_pts))
    return ratio


def register2Fragments(id1, id2, pcdpath, keyptspath, ppfdescpath):
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    # 1. load point cloud, keypoints, descriptors
    original_source_pc = get_pcd(pcdpath, cloud_bin_s)
    original_target_pc = get_pcd(pcdpath, cloud_bin_s)
    print(original_source_pc)
    print(original_target_pc)
    # downsample and estimate the normals
    voxel_size = 0.02
    source_pc = open3d.geometry.voxel_down_sample(original_source_pc, voxel_size)
    open3d.geometry.estimate_normals(source_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_pc = open3d.geometry.voxel_down_sample(original_target_pc, voxel_size)
    open3d.geometry.estimate_normals(target_pc, open3d.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    print(source_pc)
    print(target_pc)
    # load keypoints and descriptors
    source_keypts = get_keypts(keyptspath, cloud_bin_s)
    target_keypts = get_keypts(keyptspath, cloud_bin_t)
    # print(source_keypts.shape)
    source_desc = get_desc(ppfdescpath, cloud_bin_s)
    target_desc = get_desc(ppfdescpath, cloud_bin_t)

    # 2. ransac
    ransac_result = ransac_based_on_feature_matching(source_keypts, target_keypts, source_desc, target_desc)
    print("RANSAC Correspondence_set:", len(ransac_result.correspondence_set))

    # 3. refine with ICP
    icp_result = icp_refine(source_pc, target_pc, ransac_result.transformation, voxel_size * 0.4)
    print("ICP Correspondence_set:", len(ransac_result.correspondence_set))

    # 4. transform the target_pc, so that source and target are in the same coordinates.
    target_pc.transform(icp_result.transformation)

    # 5. compute the alignment
    align = cal_alignment(original_source_pc, original_target_pc, 0.05)

    print("align:", align)
    print("trans:", icp_result.transformation)


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
        ppfdescpath = os.path.join(interpath, "3dmatch_desc/")

        register2Fragments(0, 1, pcdpath, keyptspath, ppfdescpath)
        break
        # print(source_desc.shape)

        # trans = ransac_based_on_correspondence(source_keypts[0:1000], target_keypts[0:1000], source_desc[0:1000], target_desc[0:1000])
        # corr = calculate_M(source_desc[0:500], target_desc[0:500])
        # M = len(corr)
        # # TODO: 需要先用ground truth的transformation对source进行变换
        # corr_gnd = calculate_M_gnd(corr, source_keypts[0:500], target_keypts[0:500], 0.3)
        # M_gnd = len(corr_gnd)

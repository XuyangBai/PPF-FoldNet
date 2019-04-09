import open3d
import numpy as np
import time
from dataloader import get_train_loader
from input_preparation import rgbd_to_point_cloud
from scipy.spatial import KDTree


def calculate_M(source, target):
    # shape of source and target: [2048, 512]
    res = {}
    for fea1, i in zip(source, range(len(source))):
        kdtree_s = KDTree(target)
        dis, idx1 = kdtree_s.query(fea1, 1)
        fea2 = target[idx1]
        kdtree_t = KDTree(source)
        dis, idx2 = kdtree_t.query(fea2, 1)
        if i == idx2:
            res[i] = idx1
    return res
    # for fea1, i in zip(source, range(len(source))):
    #     kdtree_s = open3d.geometry.KDTreeFlann(target)
    #     _, idx1, _ = kdtree_s.search_knn_vector_xd(fea1, 1)
    #     fea2 = target[idx1]
    #
    #     kdtree_t = open3d.geometry.KDTreeFlann(source)
    #     _, idx2, _ = kdtree_t.search_knn_vector_xd(fea2, 1)
    #     if i == idx2:
    #         res[i] = idx1
    # return res


def distance(p1, p2):
    return np.sqrt(np.dot(p1 - p2, p1 - p2))


def calculate_M_gnd(matching, P, Q, tao1):
    res = {}
    for i, j in matching.items():
        if distance(P[i], Q[j]) <= tao1:
            res[i] = j
    return res


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


def evaluate(loader, tao1=0.3, tao2=0.05):
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
        # g_P = model.encoder(P)
        # g_Q = model.encoder(Q)
        g_P = P
        g_Q = Q
        matching_points = calculate_M(g_P, g_Q)
        gt_matching_points = calculate_M_gnd(matching_points, P, Q, tao1)
        if len(gt_matching_points) * 1.0 / len(matching_points) > tao2:
            pred_matching_fragments += 1
    recall = pred_matching_fragments * 1.0 / total_matching_fragments
    print(f"Recall with tao1 = {tao1*100}cm and tao2 = {tao2} is: {recall}")


if __name__ == '__main__':
    # data_dir = "data/train/sun3d-harvard_c11-hv_c11_2/seq-01-train"
    # pcd1 = rgbd_to_point_cloud(data_dir, '000002')
    # pcd2 = rgbd_to_point_cloud(data_dir, '000003')
    # print(is_matching_pairs(pcd1, pcd2))
    # res = calculate_M(np.asarray(pcd1.points)[0:20], np.asarray(pcd2.points)[0:20])
    # print(res)

    dataroot = "./data/train/sun3d-harvard_c11-hv_c11_2/seq-01-train-processed/"
    trainloader = get_train_loader(dataroot, batch_size=2)
    for iter, (patches, ids) in enumerate(trainloader):
        # TODO: 现在是同一个scene下的两个不同的fragments, 因为sample是随机的没有判成是True
        print(ids)
        pcd1 = open3d.read_point_cloud(dataroot + ids[0] + ".pcd")
        pcd2 = open3d.read_point_cloud(dataroot + ids[1] + ".pcd")
        # TODO: 这个threshold应该设多少？
        is_matching_pairs(pcd1, pcd2, threshold=0.15)
        stime = time.time()
        res = calculate_M(patches[0].reshape([-1, 512]), patches[1].reshape([-1, 512]))
        print("time:", time.time() - stime)
        print(len(res))
        break

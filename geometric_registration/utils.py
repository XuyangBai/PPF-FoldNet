import os
import open3d
import numpy as np


def get_pcd(pcdpath, filename):
    return open3d.read_point_cloud(os.path.join(pcdpath, filename + '.ply'))


def get_keypts(keyptspath, filename):
    keypts = np.fromfile(os.path.join(keyptspath, filename + '.keypts.bin'), dtype=np.float32)
    num_keypts = int(keypts[0])
    keypts = keypts[1:].reshape([num_keypts, 3])
    return keypts


def get_desc(descpath, filename, desc_name):
    if desc_name == '3dmatch':
        desc = np.fromfile(os.path.join(descpath, filename + '.desc.3dmatch.bin'), dtype=np.float32)
        num_desc = int(desc[0])
        desc_size = int(desc[1])
        desc = desc[2:].reshape([num_desc, desc_size])
    elif desc_name == 'ppf':
        desc = np.load(os.path.join(descpath, filename + '.desc.ppf.bin.npy'))
    else:
        print("No such descriptor")
        exit(-1)
    return desc


def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result


if __name__ == '__main__':
    # gtpath = './result/'
    # gtlog = loadlog(gtpath)
    # print(gtlog.keys())
    # print(gtlog['0_1'])

    a = get_desc("./intermediate-files-real/7-scenes-redkitchen/3dmatch_desc/", "cloud_bin_0", '3dmatch')
    b = get_desc("./intermediate-files-real/7-scenes-redkitchen/3dmatch_desc/", "cloud_bin_1", '3dmatch')
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    c = get_desc("./ppf_desc_04301124/7-scenes-redkitchen/", "cloud_bin_0", 'ppf')
    d = get_desc("./ppf_desc_04301124/7-scenes-redkitchen/", "cloud_bin_1", 'ppf')
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    from geometric_registration.evaluate_ppfnet import calculate_M
    import time

    start_time = time.time()
    calculate_M(a, b)
    print(time.time() - start_time)
    start_time = time.time()
    calculate_M(c, d)
    print(time.time() - start_time)

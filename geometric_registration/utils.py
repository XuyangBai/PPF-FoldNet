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


def get_desc(descpath, filename):
    #desc = np.fromfile(os.path.join(descpath, filename + '.desc.3dmatch.bin'), dtype=np.float32)
    #num_desc = int(desc[0])
    #desc_size = int(desc[1])
    #desc = desc[2:].reshape([num_desc, desc_size])
    #return desc
    desc = np.load(os.path.join(descpath, filename + '.desc.ppf.bin.npy'))
    return desc
    # if filename.__contains__('ppf'):
    #     desc = np.load(filename)
    # elif filename.__contains__('3dmatch'):
    #     desc = np.fromfile(os.path.join(descpath, filename), dtype=np.float32)
    #     num_desc = int(desc[0])
    #     desc_size = int(desc[1])
    #     desc = desc[2:].reshape([num_desc, desc_size])
    # return desc

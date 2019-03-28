from dataset import ShapeNetDataset
import numpy as np
import random
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d


def draw_pts(pts, clr, cmap, ax=None, sz=20):
    if ax is None:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        ax.view_init(-45, -64)
    else:
        ax.cla()

    ax.set_alpha(255)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim, max_lim)
    ax.set_ylim3d(min_lim, max_lim)
    ax.set_zlim3d(min_lim, max_lim)

    if cmap is None and clr is not None:
        assert (np.all(clr.shape == pts.shape))
        sct = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            edgecolors=(0.5, 0.5, 0.5)
        )

    else:
        if clr is None:
            M = ax.get_proj()
            _, clr, _ = proj3d.proj_trans_points(pts, M)
            # _, clr, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], M)
        clr = (clr - clr.min()) / (clr.max() - clr.min())  # normalization
        sct = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            cmap=cmap,
            # depthshade=False,
            edgecolors=(0.5, 0.5, 0.5)
        )

    ax.set_axis_off()
    ax.set_facecolor("white")
    return ax, sct


if __name__ == '__main__':
    dataroot = "data/shapenetcore_partanno_segmentation_benchmark_v0"
    dataset = ShapeNetDataset(root=dataroot,
                              class_choice='Airplane',
                              split='train',
                              classification=True,
                              num_points=2048,
                              )
    ax, sct = draw_pts(dataset[random.randint(0, 100)][0], clr=None, cmap='CMRmap')
    ax.figure.show()

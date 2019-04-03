from dataset import ShapeNetDataset
from dataloader import get_dataloader
from matplotlib import pyplot as plt
from visualize import draw_pts
from model import FoldNet
import torch
import random


def show_reconstructed(model, class_choice='Airplane'):
    dataroot = "data/shapenetcore_partanno_segmentation_benchmark_v0"

    dataloader = get_dataloader(root=dataroot,
                                split='test',
                                class_choice=class_choice,
                                classification=True,
                                num_points=2048,
                                shuffle=False
                                )
    pts, _ = dataloader.dataset[random.randint(0, 100)]
    reconstructed_pl = model(pts.view(1, 2048, 3))[0]
    ax1, _ = draw_pts(pts, clr=None, cmap='CMRmap')
    ax2, _ = draw_pts(reconstructed_pl.detach().numpy(), clr=None, cmap='CMRmap')
    ax1.figure.show()
    ax2.figure.show()
    # fig = plt.figure()
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    # draw_pts(pts, clr=None, cmap='CMRmap', ax=ax1)
    # draw_pts(reconstructed_pl.detach().numpy(), clr=None, cmap='CMRmap', ax=ax2)
    # fig.add_subplot(ax1)
    # fig.add_subplot(ax2)
    # plt.show()


def interpolate(model, class1='Airplane', class2=None):
    dataroot = "data/shapenetcore_partanno_segmentation_benchmark_v0"
    dataset1 = ShapeNetDataset(root=dataroot,
                               class_choice=class1,
                               split='test',
                               classification=True,
                               num_points=2048,
                               )
    pts1, _ = dataset1[random.randint(0, 100)]
    codeword1 = model.encoder(pts1.view(1, 2048, 3))
    # intra-class or inter-class
    if class2 is not None:
        dataset2 = ShapeNetDataset(root=dataroot,
                                   class_choice=class2,
                                   split='test',
                                   classification=True,
                                   num_points=2048)
        pts2, _ = dataset2[random.randint(0, 100)]
        codeword2 = model.encoder(pts2.view(1, 2048, 3))
    else:
        pts2, _ = dataset1[random.randint(0, 100)]
        codeword2 = model.encoder(pts2.view(1, 2048, 3))

    # do interpolation.
    ratio = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # ratio = [0, 1]
    for u in range(len(ratio)):
        mix_codeword1 = (1 - ratio[u]) * codeword1 + ratio[u] * codeword2
        output = model.decoder(mix_codeword1)
        plt.subplot(1, u + 1, 1)
        pts = output[0].detach().numpy()
        ax, _ = draw_pts(pts, clr=None, cmap='CMRmap')
        ax.figure.show()


if __name__ == '__main__':
    # pretrain = 'models/model.pkl'
    pretrain = 'models/model_nos_noa.pkl'
    model = FoldNet(num_points=2048)
    state_dict = torch.load(pretrain, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    show_reconstructed(model, 'Airplane')
    # interpolate(model, "Airplane", "Lamp")

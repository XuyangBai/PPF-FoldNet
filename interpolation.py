from dataset import ShapeNetDataset
from matplotlib import pyplot as plt
from visualize import draw_pts
from model import FoldNet


def interpolate(model, class1='Airplane', class2=None):
    dataroot = "data/shapenetcore_partanno_segmentation_benchmark_v0"
    dataset1 = ShapeNetDataset(root=dataroot,
                               class_choice=class1,
                               split='train',
                               classification=True,
                               num_points=2048,
                               )
    codeword1 = model.encoder(dataset1[0])
    # intra-class or inter-class
    if class2 is None:
        dataset2 = ShapeNetDataset(root=dataroot,
                                   class_choice=class2,
                                   split='train',
                                   classification=True,
                                   num_points=2048)
        codeword2 = model.encoder(dataset2[0])
    else:
        codeword2 = model.encoder(dataset1[1])

    # do interpolation.
    ratio = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for u in range(len(ratio)):
        mix_codeword1 = (1 - ratio[u]) * codeword1 + ratio[u] * codeword2
        output = model.decoder(mix_codeword1)
        plt.subplot(1, u, 1)
        ax, _ = draw_pts(output, clr=None, cmap='CMRmap')
        ax.figure.show()


if __name__ == '__main__':
    pretrain = ''
    model = FoldNet(num_points=2048)
    model.load_state_dict(pretrain)
    interpolate(model, "Airplane", "Table")

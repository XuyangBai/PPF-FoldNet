import torch
from dataset import SunDataset


def get_dataloader(root, split='train', batch_size=2, num_workers=4, shuffle=True):
    dataset = SunDataset(
        root=root,
        split=split,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':
    dataset = 'sun3d'
    dataroot = "./data/sun3d-harvard_c11-hv_c11_2/seq-01-test-npy"
    dataloader = get_dataloader(dataroot, batch_size=2)
    for iter, (pts) in enumerate(dataloader):
        print("points:", pts.shape)
        pts = pts.reshape([-1, pts.shape[2], pts.shape[3]])
        print("points after reshape:", pts.shape)
        break

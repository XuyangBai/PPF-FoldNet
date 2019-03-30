import torch
from dataset import ShapeNetDataset


def get_dataloader(root, split='train', class_choice=None, classification=True, batch_size=32, num_points=2048,
                   num_workers=4, shuffle=True):
    dataset = ShapeNetDataset(
        root=root,
        split=split,
        class_choice=class_choice,
        num_points=num_points,
        classification=classification,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':
    dataset = "shapenet"
    dataroot = "data/shapenetcore_partanno_segmentation_benchmark_v0"
    dataloader = get_dataloader(dataroot, batch_size=4, num_points=2048)
    print("dataloader size:", dataloader.dataset.__len__())
    for iter, (pts, seg) in enumerate(dataloader):
        print("points:", pts.shape, pts.type)
        print("segs  :", seg.shape, seg.type)
        break

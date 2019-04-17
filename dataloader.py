import torch
import time
from dataset import SunDataset


def get_dataloader(root, split, batch_size=1, num_patches=32, num_points_per_patch=1024, num_workers=4, shuffle=True,
                   on_the_fly=True):
    dataset = SunDataset(
        root=root,
        split=split,
        num_patches=num_patches,
        num_points_per_patch=num_points_per_patch,
        on_the_fly=on_the_fly
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


# def get_test_loader(root, num_workers=4):
#     dataset = SunDataset(
#         root=root,
#         split='test',
#     )
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=True,
#         num_workers=num_workers
#     )
#     return dataloader


if __name__ == '__main__':
    dataset = 'sun3d'
    dataroot = "./data/train/sun3d-harvard_c11-hv_c11_2/seq-01-train"
    trainloader = get_dataloader(dataroot, split='train', batch_size=1)
    start_time = time.time()
    for iter, (patches, ids) in enumerate(trainloader):
        print("patches:", patches.shape)
        # print("ids:", ids)
        if iter == 9:
            break
    print(f"On the fly: {time.time() - start_time}")

    dataroot = "./data/train/sun3d-harvard_c11-hv_c11_2/seq-01-train-processed"
    trainloader = get_dataloader(dataroot, split='train', batch_size=1, on_the_fly=False)
    start_time = time.time()
    for iter, (patches, ids) in enumerate(trainloader):
        print("patches:", patches.shape)
        if iter == 9:
            break
    print(f"Not on the fly: {time.time() - start_time}")

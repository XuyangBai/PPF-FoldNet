import torch
from dataset import SunDataset


def get_dataloader(root, split, batch_size=2, num_patches=32, num_workers=4, shuffle=True):
    dataset = SunDataset(
        root=root,
        split=split,
        num_patches=num_patches
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
    dataroot = "./data/train/sun3d-harvard_c11-hv_c11_2/seq-01-train-processed"
    trainloader = get_dataloader(dataroot, split='train', batch_size=4)
    for iter, (patches, ids) in enumerate(trainloader):
        print("patches:", patches.shape)
        print("ids:", ids)
        patches = patches.reshape([-1, patches.shape[2], patches.shape[3]])
        print("patches after reshape:", patches.shape)
        break

    # testloader = get_test_loader(dataroot)
    # for iter, (patches, pcd) in enumerate(testloader):
    #     print("patches:", patches.shape)
    #     print("point cloud")
    #     print(pcd)

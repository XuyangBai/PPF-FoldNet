import time
from dataset import SunDataset
import torch


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


if __name__ == '__main__':
    dataset = 'sun3d'
    dataroot = "/data/3DMatch/whole"
    trainloader = get_dataloader(dataroot, split='test', batch_size=32)
    start_time = time.time()
    print(f"Totally {len(trainloader)} iter.")
    for iter, (patches, ids) in enumerate(trainloader):
        if iter % 100 == 0:
            print(f"Iter {iter}: {time.time() - start_time} s")
    print(f"On the fly: {time.time() - start_time}")

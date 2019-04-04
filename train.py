import os
import time
import shutil
from torch import optim
from trainer import Trainer
from model import FoldNet
from dataloader import get_dataloader


class Args(object):
    def __init__(self):
        self.experiment_id = "FoldNet" + time.strftime('%m%d%H%M')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)
        shutil.copy2(os.path.join('.', 'train.py'), os.path.join(snapshot_root, 'train.py'))
        self.epoch = 100
        self.num_points = 2048
        self.batch_size = 16
        self.dataset = 'shapenet'
        self.data_dir = 'data/shapenetcore_partanno_segmentation_benchmark_v0'

        self.gpu_mode = False
        self.verbose = False

        # model & optimizer
        self.model = FoldNet(self.num_points)
        self.pretrain = ''
        self.parameter = self.model.get_parameter()
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.scheduler_interval = 100

        # dataloader
        self.train_loader = get_dataloader(root=self.data_dir,
                                           split='train',
                                           classification=True,
                                           batch_size=self.batch_size,
                                           num_points=self.num_points,
                                           shuffle=False
                                           )
        self.test_loader = get_dataloader(root=self.data_dir,
                                          split='test',
                                          classification=True,  # if True then return pts & cls
                                          batch_size=self.batch_size,
                                          num_points=self.num_points,
                                          shuffle=False
                                          )
        print("Training set size:", self.train_loader.dataset.__len__())
        print("Test set size:", self.test_loader.dataset.__len__())

        # snapshot
        self.snapshot_interval = 10
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = tensorboard_root

        self.check_args()

    def check_args(self):
        """checking arguments"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.tboard_dir):
            os.makedirs(self.tboard_dir)
        return self


if __name__ == '__main__':
    args = Args()
    trainer = Trainer(args)
    trainer.train()

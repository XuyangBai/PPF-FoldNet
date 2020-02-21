import os
import time
import shutil
from dataloader import get_dataloader
from trainer import Trainer
from loss.chamfer_loss import ChamferLoss
from models.model_conv1d import PPFFoldNet
from torch import optim


class Args(object):
    def __init__(self):
        self.experiment_id = "PPF-FoldNet" + time.strftime('%m%d%H%M')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)
        shutil.copy2(os.path.join('.', 'train.py'), os.path.join(snapshot_root, 'train.py'))
        shutil.copy2(os.path.join('.', 'models/model_conv1d.py'), os.path.join(snapshot_root, 'model.py'))
        self.epoch = 20
        self.num_patches = 1
        self.num_points_per_patch = 1024  # num of points per patches
        self.batch_size = 32
        self.dataset = 'sun3d'
        self.data_train_dir = './data/3DMatch/rgbd_fragments'
        self.data_test_dir = './data/3DMatch/rgbd_fragments'

        self.gpu_mode = True
        self.verbose = True

        # model & optimizer
        self.model = PPFFoldNet(self.num_patches, self.num_points_per_patch)
        self.pretrain = ''
        self.parameter = self.model.get_parameter()
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.scheduler_interval = 10

        # dataloader
        self.train_loader = get_dataloader(root=self.data_train_dir,
                                                      batch_size=self.batch_size,
                                                      split='train',
                                                      num_patches=self.num_patches,
                                                      num_points_per_patch=self.num_points_per_patch,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      )
        self.test_loader = get_dataloader(root=self.data_test_dir,
                                                     batch_size=self.batch_size,
                                                     split='test',
                                                     num_patches=self.num_patches,
                                                     num_points_per_patch=self.num_points_per_patch,
                                                     shuffle=False,
                                                     num_workers=4,
                                                     )
        print("Training set size:", self.train_loader.dataset.__len__())
        print("Test set size:", self.test_loader.dataset.__len__())
        # snapshot
        self.snapshot_interval = 100000
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = tensorboard_root

        # evaluate
        self.evaluate_interval = 2
        self.evaluate_metric = ChamferLoss()

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

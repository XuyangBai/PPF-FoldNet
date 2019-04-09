import os
import time
import shutil
from torch import optim
from trainer import Trainer
from model import PPFFoldNet
from dataloader import get_dataloader


class Args(object):
    def __init__(self):
        self.experiment_id = "PPF-FoldNet" + time.strftime('%m%d%H%M')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)
        shutil.copy2(os.path.join('.', 'train.py'), os.path.join(snapshot_root, 'train.py'))
        self.epoch = 300
        self.num_points = 1024  # num of points per patches
        # TODO: I do not know whether this is correct.
        #  I pick all the local patches from one point cloud fragment
        #  So the input to the network is [bs, 2048, 1024, 4]
        self.batch_size = 2
        self.dataset = 'sun3d'
        self.data_train_dir = './data/train/sun3d-harvard_c11-hv_c11_2/seq-01-train-processed'
        self.data_test_dir = './data/train/sun3d-harvard_c11-hv_c11_2/seq-01-test-processed'

        self.gpu_mode = False
        self.verbose = False

        # model & optimizer
        self.model = PPFFoldNet(self.num_points)
        self.pretrain = ''
        self.parameter = self.model.get_parameter()
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.scheduler_interval = 100

        # dataloader
        self.train_loader = get_dataloader(root=self.data_train_dir,
                                           batch_size=self.batch_size,
                                           split='train',
                                           shuffle=False
                                           )
        self.test_loader = get_dataloader(root=self.data_test_dir,
                                          batch_size=2,  # batch size of test loader have to be 2
                                          split='test',
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

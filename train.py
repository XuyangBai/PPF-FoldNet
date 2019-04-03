import os
import time
import shutil
from trainer import Trainer


class Args(object):
    def __init__(self):
        self.experiment_id = "FoldNet" + time.strftime('%m%d%H%M')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        os.makedirs(snapshot_root, exist_ok=True)
        shutil.copy2(os.path.join('.', 'train.py'), os.path.join(snapshot_root, 'train.py'))
        self.epoch = 100
        self.num_points = 2048
        self.batch_size = 16
        self.dataset = 'shapenet'
        self.data_dir = 'data/'
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = os.path.join(snapshot_root, 'tboard/')
        self.gpu_mode = False
        self.verbose = False
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 1e-6

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

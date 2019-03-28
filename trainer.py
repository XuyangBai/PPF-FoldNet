import torch
import torch.optim as optim
import time, os
import numpy as np
from model import FoldNet
from dataloader import get_dataloader
from visualize import draw_pts


class Trainer(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.num_points = args.num_points
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        if self.dataset == 'shapenet':
            self.data_dir = os.path.join(args.data_dir, 'shapenetcore_partanno_segmentation_benchmark_v0')
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose

        self.model = FoldNet(args.num_points)
        self.parameter = self.model.get_parameter()
        self.optimizer = optim.Adam(self.parameter, lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        self.train_loader = get_dataloader(root=self.data_dir,
                                           split='train',
                                           classification=True,
                                           batch_size=self.batch_size,
                                           num_points=self.num_points)
        self.test_loader = get_dataloader(root=self.data_dir,
                                          split='test',
                                          classification=True,  # if True then return pts & cls
                                          batch_size=self.batch_size,
                                          num_points=self.num_points)
        print("Training set size:", self.train_loader.dataset.__len__())
        print("Test set size:", self.test_loader.dataset.__len__())

        if self.gpu_mode:
            self.model = self.model.cuda()

    def train(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print('training start!!')
        start_time = time.time()

        self.model.train()
        for epoch in range(self.epoch):
            self.train_epoch(epoch, self.verbose)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                if res['loss'] < best_loss:
                    best_loss = res['loss']
                    self._snapshot('best')

            if (epoch + 1) % 10 == 0:
                self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                self._snapshot(epoch + 1)

        # finish all epoch
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_epoch(self, epoch, verbose=False):
        epoch_start_time = time.time()
        loss_buf = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        for iter, (pts, _) in enumerate(self.train_loader):
            if self.gpu_mode:
                pts = pts.cuda()
            # forward
            self.optimizer.zero_grad()
            output = self.model(pts)
            loss = self.model.get_loss(pts, output)
            # backward
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())

            if (iter + 1) % 10 == 0 and self.verbose:
                print(
                    f"Epoch: {epoch+1} [{iter+1:4d}/{num_batch}] loss: {loss:.2f} time: {time.time() - epoch_start_time:.2f}s")
        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')

    def evaluate(self, epoch):
        self.model.eval()
        loss_buf = []
        for iter, (pts, _) in enumerate(self.train_loader):
            if self.gpu_mode:
                pts = pts.cuda()
            output = self.model(pts)
            loss = self.model.get_loss(pts, output)
            loss_buf.append(loss.detach().cpu().numpy())

        # show the reconstructed image from train set
        pts, _ = self.train_loader.dataset[0]
        if self.gpu_mode:
            pts = pts.cuda()
        reconstructed_pl = self.model(pts.view(1, 2048, 3))[0]
        ax1, _ = draw_pts(pts.cpu().detach().numpy(), clr=None, cmap='CMRmap')
        ax2, _ = draw_pts(reconstructed_pl.cpu().detach().numpy(), clr=None, cmap='CMRmap')
        ax2.figure.savefig(self.result_dir + 'train_' + str(epoch) + ".png")
        if epoch == 0:
            ax1.figrue.savefig(self.result_dir + 'train_input.png')
        # show the reconstructed image from test set
        pts, _ = self.test_loader.dataset[0]
        if self.gpu_mode:
            pts = pts.cuda()
        reconstructed_pl = self.model(pts.view(1, 2048, 3))[0]
        ax1, _ = draw_pts(pts.cpu().detach().numpy(), clr=None, cmap='CMRmap')
        ax2, _ = draw_pts(reconstructed_pl.cpu().detach().numpy(), clr=None, cmap='CMRmap')
        ax2.figure.savefig(self.result_dir + 'test_' + str(epoch) + ".png")
        if epoch == 0:
            ax1.figrue.savefig(self.result_dir + 'test_input.png')

        self.model.train()
        res = {
            'loss': np.mean(loss_buf),
        }
        return res

    def _snapshot(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset)
        torch.save(self.model.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")

    def _load_pretrain(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset)
        self.model.load(self.model.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
        print(f"Load model from {save_dir}_{str(epoch)}.pkl")

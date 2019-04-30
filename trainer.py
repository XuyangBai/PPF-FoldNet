import torch
import torch.optim as optim
import time, os
import numpy as np
from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.num_points_per_patch = args.num_points_per_patch
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        if self.dataset == 'shapenet':
            self.data_dir = os.path.join(args.data_dir, 'shapenetcore_partanno_segmentation_benchmark_v0')
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.evaluate_interval = args.evaluate_interval
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        
        self.train_loader = args.train_loader
        self.test_loader = args.test_loader

        if self.gpu_mode:
            self.model = self.model.cuda()

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

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
        res = self.evaluate(0)
        print(f'Evaluation: Epoch 0: Loss {res["loss"]}')
        for epoch in range(self.epoch):
            self.train_epoch(epoch)

            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                print(f'Evaluation: Epoch {epoch+1}: Loss {res["loss"]}')
                if res['loss'] < best_loss:
                    best_loss = res['loss']
                    self._snapshot('best')
                if self.writer:
                    self.writer.add_scalar('Loss', res['loss'], epoch)
             
            if epoch % self.scheduler_interval == 0:
                self.scheduler.step()
            
            if epoch % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

            if self.writer:
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)

                # finish all epoch
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        for iter, (patches, ids) in enumerate(self.train_loader):
            patches = patches.reshape([-1, patches.shape[2], patches.shape[3]])
            if self.gpu_mode:
                patches = patches.cuda()
            # forward
            self.optimizer.zero_grad()
            output = self.model(patches)
            loss = self.model.get_loss(patches, output)
            # backward
            loss.backward()
            self.optimizer.step()
            # loss_buf.append(loss.detach().cpu().numpy())
            loss_buf.append(float(loss))

            if (iter + 1) % 100 == 0 and self.verbose:
                iter_time = time.time() - epoch_start_time
                print(f"Epoch: {epoch+1} [{iter+1:4d}/{num_batch}] loss: {loss:.2f} time: {iter_time:.2f}s")
            del loss
            del output
        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')
        del loss_buf

    def evaluate(self, epoch):
        self.model.eval()
        loss_buf = []
        for iter, (patches, ids) in enumerate(self.test_loader):
            patches = patches.reshape([-1, patches.shape[2], patches.shape[3]])
            if self.gpu_mode:
                patches = patches.cuda()
            output = self.model(patches)
            loss = self.model.get_loss(patches, output)
            loss_buf.append(float(loss))
            del loss
            del output
        self.model.train()
        res = {
            'loss': np.mean(loss_buf),
        }
        del loss_buf
        return res

    def _snapshot(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset)
        torch.save(self.model.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Load model from {pretrain}.pkl")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

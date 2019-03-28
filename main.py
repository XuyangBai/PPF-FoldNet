import numpy as np
import argparse, os
import torch
from trainer import Trainer

torch.manual_seed(1)
np.random.seed(1)


def parse_args():
    """parsing and configuration"""
    desc = "Program Entry for PPFFold-Net"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--num_points', type=int, default=2048, help='The size of point cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet', 'modelnet'])
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory name to find the dataset')
    parser.add_argument('--save_dir', type=str, default='models/', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results/', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--gpu_mode', type=bool, default=False, help='whether use GPU')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    return check_args(parser.parse_args())


def check_args(args):
    """checking arguments"""
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

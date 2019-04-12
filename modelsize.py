import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512) # [bs, 1024, 512]

        # self.maxpool = nn.MaxPool1d(1024) # [bs, 1, 512]

        # self.conv4 = nn.Conv1d(512, 256, 1)
        # self.relu4 = nn.ReLU()
        # self.conv5 = nn.Conv1d(256, 64, 1)
        # self.relu5 = nn.ReLU()
        # self.conv6 = nn.Conv1d(64, 4, 1)
        #
        # self.conv7 = nn.Conv1d(516, 256, 1)
        # self.relu6 = nn.ReLU()
        # self.conv8 = nn.Conv1d(256, 64, 1)
        # self.relu7 = nn.ReLU()
        # self.conv9 = nn.Conv1d(64, 4, 1)

        # print(list(self.modules()))


class SizeEstimator(object):

    def __init__(self, model, input_size=(1, 1, 32, 32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = 32
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        with torch.no_grad():
            input_ = Variable(torch.FloatTensor(*self.input_size))
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            # print(bits * 2 / 8 / 1024 / 1024)
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits * 2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        # param = [np.prod(list(p.size())) for p in self.model.parameters()]
        # print(sum(param) / 1024 / 1024)
        print(f"Param: {self.param_bits/8/1024/1024} MB")
        print(f"Input: {self.input_bits/8/1024/1024} MB")
        print(f"Intermedia: {self.forward_backward_bits/8/1024/1024} MB")
        total_megabytes = (total / 8) / (1024 ** 2)
        return total_megabytes, total


if __name__ == '__main__':
    se = SizeEstimator(Model(), input_size=(10, 4, 1024), bits=32)
    res = se.estimate_size()
    print(res)

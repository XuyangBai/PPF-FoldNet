import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, num_patches=32, num_points_per_patch=1024):
        super(Encoder, self).__init__()
        self.num_patches = num_patches
        self.num_points_per_patches = num_points_per_patch
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(704, 1024)
        self.fc2 = nn.Linear(1024, 512)  # codeword dimension = 512
        # self.bn4 = nn.BatchNorm1d(1024)
        # self.bn5 = nn.BatchNorm1d(512)

    def forward(self, input):
        # origin shape from dataloader [bs*32, 1024(num_points), 4]
        input = input.transpose(-1, -2).float()
        x = self.relu1(self.bn1(self.conv1(input)))
        local_feature_1 = x  # save the  low level features to concatenate this global feature.
        x = self.relu2(self.bn2(self.conv2(x)))
        local_feature_2 = x
        x = self.relu3(self.bn3(self.conv3(x)))
        local_feature_3 = x
        # max at the third dimension, which means that for one local patch, choose the largest feature.
        x = torch.max(x, 2, keepdim=True)[0]
        global_feature = x.repeat([1, 1, self.num_points_per_patches])
        # feature shape: [num_patches, 704, num_points_per_patch]
        feature = torch.cat([local_feature_1, local_feature_2, local_feature_3, global_feature], 1)

        # TODO: add batch_norm or not?
        x = F.relu(self.fc1(feature.transpose(1, 2)))
        x = F.relu(self.fc2(x))

        # still max at the second dimension.
        return torch.max(x, 1, keepdim=True)[0]  # [bs, 1, 512]


class Decoder(nn.Module):
    def __init__(self, num_points_per_patch=1024):
        super(Decoder, self).__init__()
        self.m = num_points_per_patch  # 32 * 32
        self.meshgrid = [[0, 1, 32], [0, 1, 32]]
        self.mlp1 = nn.Sequential(
            nn.Conv1d(514, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 4, 1),
            # nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(516, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 4, 1),
            # nn.ReLU(),
        )

    def build_grid(self, batch_size):
        # ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in self.meshgrid])
        # ndim = len(self.meshgrid)
        # grid = np.zeros((np.prod([it[2] for it in self.meshgrid]), ndim), dtype=np.float32)  # MxD
        # for d in range(ndim):
        #     grid[:, d] = np.reshape(ret[d], -1)
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        grid = np.array(list(itertools.product(x, y)))
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
        grid = torch.tensor(grid)
        return grid.float()

    def forward(self, input):
        input = input.transpose(1, 2).repeat(1, 1, self.m)  # [bs, 512, m]
        grid = self.build_grid(input.shape[0]).transpose(1, 2)  # [bs, 2, m]
        if torch.cuda.is_available():
            grid = grid.cuda()
        concate1 = torch.cat((input, grid), dim=1)  # [bs, 514, m]
        after_folding1 = self.mlp1(concate1)  # [bs, 4, m]
        concate2 = torch.cat((input, after_folding1), dim=1)  # [bs, 516, m]
        after_folding2 = self.mlp2(concate2)  # [bs, 4, m]
        return after_folding2.transpose(1, 2)  # [bs, m, 4]


class PPFFoldNet(nn.Module):
    """
    1. skip connection scheme.
    2. use linear / conv1d.
    3. whether to use relu in last layer.
    """

    def __init__(self, num_patches=32, num_points_per_patch=1024):
        super(PPFFoldNet, self).__init__()
        self.encoder = Encoder(num_patches=num_patches, num_points_per_patch=num_points_per_patch)
        self.decoder = Decoder(num_points_per_patch=num_points_per_patch)

        # Print the params size of this model.
        if torch.cuda.is_available():
            summary(self.cuda(), (1024, 4), batch_size=num_patches)
        else:
            summary(self, (1024, 4), batch_size=num_patches)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        codeword = self.encoder(input.float())
        output = self.decoder(codeword)
        return output

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


if __name__ == '__main__':
    input = torch.rand(32, 1024, 4)
    model = PPFFoldNet(num_patches=32, num_points_per_patch=1024)
    output = model(input)
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)

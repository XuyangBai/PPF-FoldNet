import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from loss import ChamferLoss
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, num_patches=32, num_points_per_patch=1024):
        super(Encoder, self).__init__()
        self.num_patches = num_patches
        self.num_points_per_patches = num_points_per_patch
        self.conv1 = nn.Linear(4, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(64 + 256, 512)
        self.fc2 = nn.Linear(512, 512)  # codeword dimension = 512
        # self.bn4 = nn.BatchNorm1d(1024)
        # self.bn5 = nn.BatchNorm1d(512)

    def forward(self, input):
        # origin shape from dataloader [bs*32, 1024(num_points), 4]
        # input = input.transpose(-1, -2).float()
        x = self.relu1(self.bn1(self.conv1(input).transpose(-1, -2)))
        local_feature_1 = x.transpose(-1, -2)  # save the  low level features to concatenate this global feature.
        x = self.relu2(self.bn2(self.conv2(x.transpose(-1, -2)).transpose(-1, -2)))
        local_feature_2 = x.transpose(-1, -2)
        x = self.relu3(self.bn3(self.conv3(x.transpose(-1, -2)).transpose(-1, -2)))
        local_feature_3 = x.transpose(-1, -2)
        # TODO: max at the second dimension, which means that for one local patch, choose the largest feature.
        x = x.transpose(-1, -2)
        x = torch.max(x, 1, keepdim=True)[0]
        global_feature = x.repeat([1, self.num_points_per_patches, 1])
        # feature shape: [num_patches, num_points_per_patch, 704]
        feature = torch.cat([local_feature_1, global_feature], -1)

        # TODO: add batch_norm or not?
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))

        # TODO: still max at the second dimension.
        return torch.max(x, 1, keepdim=True)[0]  # [bs, 1, 512]


class Decoder(nn.Module):
    def __init__(self, num_points_per_patch=1024):
        super(Decoder, self).__init__()
        self.m = num_points_per_patch  # 32 * 32
        self.meshgrid = [[-0.3, 0.3, 32], [-0.3, 0.3, 32]]
        self.mlp1 = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            # nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(516, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
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
        input = input.repeat(1, self.m, 1)  # [bs, m, 512]
        grid = self.build_grid(input.shape[0])  # [bs, m, 2]
        if torch.cuda.is_available():
            grid = grid.cuda()
        concate1 = torch.cat((input, grid), dim=-1)  # [bs, m, 514]
        after_folding1 = self.mlp1(concate1)  # [bs, m, 4]
        concate2 = torch.cat((input, after_folding1), dim=-1)  # [bs, m, 516]
        after_folding2 = self.mlp2(concate2)  # [bs, m, 4]
        return after_folding2  # [bs, m, 4]


class PPFFoldNet(nn.Module):
    """
    This model is similar with PPFFoldNet defined in model.py, the difference is:
        1. I use Linear layer to replace Conv1d because the test shows that Linear is faster.
        2. Different skip connection scheme.
    """

    def __init__(self, num_patches=32, num_points_per_patch=1024):
        super(PPFFoldNet, self).__init__()
        self.encoder = Encoder(num_patches=num_patches, num_points_per_patch=num_points_per_patch)
        self.decoder = Decoder(num_points_per_patch=num_points_per_patch)
        self.loss = ChamferLoss()

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

    def get_loss(self, input, output):
        return self.loss(input, output)


if __name__ == '__main__':
    input = torch.rand(32, 1024, 4)
    model = PPFFoldNet(num_patches=32, num_points_per_patch=1024)
    output = model(input)
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)

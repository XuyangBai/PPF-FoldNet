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
        # TODO: max at the third dimension, which means that for one local patch, choose the largest feature.
        x = torch.max(x, 2, keepdim=True)[0]
        global_feature = x.repeat([1, 1, self.num_points_per_patches])
        # feature shape: [num_patches, 704, num_points_per_patch]
        feature = torch.cat([local_feature_1, local_feature_2, local_feature_3, global_feature], 1)

        # TODO: add batch_norm or not?
        x = F.relu(self.fc1(feature.transpose(1, 2)))
        x = self.fc2(x)

        # TODO: still max at the second dimension.
        return torch.max(x, 1, keepdim=True)[0]  # [bs, 1, 512]


class MyNet(nn.Module):
    """
    1. skip connection scheme.
    2. use linear / conv1d.
    3. whether to use relu in last layer.
    """

    def __init__(self, num_patches=32, num_points_per_patch=1024):
        super(MyNet, self).__init__()
        self.encoder = Encoder(num_patches=num_patches, num_points_per_patch=num_points_per_patch)

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
        return codeword.squeeze(dim=1)

    def get_parameter(self):
        return list(self.encoder.parameters())


if __name__ == '__main__':
    input = torch.rand(32, 1024, 4)
    model = MyNet(num_patches=32, num_points_per_patch=1024)
    output = model(input)
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)

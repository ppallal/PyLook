import torch
import torch.nn as nn
import torch.nn.functional as F

from load import config
from data_generator import DataGenerator


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(int(16 * DataGenerator.eye_res / 4 * DataGenerator.eye_res / 4), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def load(self):
        if config['cuda']:
            self.load_state_dict(torch.load('nets/net_gpu'))
        else:
            self.load_state_dict(torch.load('nets/net_cpu'))

    def save(self):
        if config['cuda']:
            torch.save(self.state_dict(), "nets/net_gpu")
            torch.save(self.cpu().state_dict(), "nets/net_cpu")
        else:
            torch.save(self.state_dict(), "nets/net_cpu")

    def package(self):
        self.save()
        # todo: code to zip the nets and backup.


net = Net()
if config['cuda']:
    net = net.cuda()

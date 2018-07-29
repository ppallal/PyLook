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
        self.conv_eye_1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv_eye_2 = nn.Conv2d(6, 16, 5, padding=2)

        self.conv_face_1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv_face_2 = nn.Conv2d(6, 16, 5, padding=2)

        self.conv_frame_1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv_frame_2 = nn.Conv2d(6, 16, 5, padding=2)

        # an affine operation: y = Wx + b
        d = (DataGenerator.eye_res / 4 * DataGenerator.eye_res / 4) + (DataGenerator.face_res / 4 * DataGenerator.face_res / 4) + (DataGenerator.frame_save_res / 4 * DataGenerator.frame_save_res / 4)

        self.out = nn.Sequential(
            nn.Linear(int(16 * d), 200),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(200, 120),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(120, 84),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(84, 2)
        )
        self.init_weights()

        # self.fc0 = nn.Linear(int(16 * d), 200)
        # self.fc1 = nn.Linear(200, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 2)

        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc2.weight)
        # torch.nn.init.xavier_uniform(self.fc3.weight)


    def forward(self, frame, face, eye):
        frame = F.max_pool2d(F.relu(self.conv_frame_1(frame)), (2, 2))
        frame = F.max_pool2d(F.relu(self.conv_frame_2(frame)), 2)

        face = F.max_pool2d(F.relu(self.conv_face_1(face)), (2, 2))
        face = F.max_pool2d(F.relu(self.conv_face_2(face)), 2)

        eye = F.max_pool2d(F.relu(self.conv_eye_1(eye)), (2, 2))
        eye = F.max_pool2d(F.relu(self.conv_eye_2(eye)), 2)

        # print(frame.size(), face.size(), [tmp.view(-1, self.num_flat_features(tmp)) for tmp in [frame, eye, face]][0].size())
        x = torch.cat([tmp.view(-1, self.num_flat_features(tmp)) for tmp in [frame, eye, face]], dim=1)
        # x = x.view(-1, self.num_flat_features(x))
        x = self.out(x)
        return x

    def init_weights(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal(m.weight)
                m.bias.data.fill_(0.01)
        self.out.apply(init_weights)
        nn.init.xavier_uniform(self.conv_eye_1.weight)
        nn.init.xavier_uniform(self.conv_eye_2.weight)
        nn.init.xavier_uniform(self.conv_face_2.weight)
        nn.init.xavier_uniform(self.conv_face_1.weight)
        nn.init.xavier_uniform(self.conv_frame_1.weight)
        nn.init.xavier_uniform(self.conv_frame_2.weight)

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

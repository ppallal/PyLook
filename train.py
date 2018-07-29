import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from load import config
from data_generator import DataGenerator, DataGenerator2
from model import net

import torch.optim as optim
import random
from tqdm import tqdm


class Trainer:

    def __init__(self, batches=1000, lr=0.01):
        self.optimizer = optim.SGD(net.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.batch_size = 8

        self.net = net
        if config['cuda']:
            self.net = self.net.cuda()

        self.n_data, self.n_lbl = DataGenerator2.load()
        self.FILT = list(range(len(self.n_lbl)))

        self.batches = batches

    # @profile
    def create_batch(self):
        sample = random.sample(self.FILT, self.batch_size)
        t = self.n_data[sample]
        t = list(zip(*t))
        frame, face, eye = np.array(t[0]) / 255, np.array(t[1]) / 255, np.array(t[2]) / 255
        # in your training loop:
        frame = torch.Tensor(frame.reshape(self.batch_size, 1, frame[0].shape[1], -1))
        face = torch.Tensor(face.reshape(self.batch_size, 1, face[0].shape[1], -1))
        eye = torch.Tensor(eye.reshape(self.batch_size, 1, eye[0].shape[1], -1))

        L = torch.Tensor(self.n_lbl[sample])
        if config['cuda']:
            frame, face, eye, L = frame.cuda(), face.cuda(), eye.cuda(), L.cuda()
        frame, face, eye = [torch.autograd.Variable(tmp, requires_grad=True) for tmp in [frame, face, eye]]
        L = torch.autograd.Variable(L, requires_grad=False)
        return (frame, face, eye), L

    # @profile
    def train(self, ax=None, overall_ax=None, refresh_rate=24):
        losses = []
        t = time.time()
        for _ in tqdm(range(self.batches)):
            self.optimizer.zero_grad()

            X, L = self.create_batch()
            Y = self.net(*X)

            loss = self.criterion(Y, L)
            l = loss.cpu() if config['cuda'] else loss
            losses.append(l.data.numpy())

            if time.time() - t > 1/refresh_rate:
                if ax is not None:
                    ax.clear()
                    ax.plot(pd.Series(losses[-1000:]).ewm(50).mean())
                if overall_ax is not None:
                    overall_ax.clear()
                    overall_ax.plot(pd.Series(losses).ewm(50).mean())
                t = time.time()

            loss.backward()
            self.optimizer.step()
        return losses


if __name__ == '__main__':
    losses = Trainer(100).train()
    pd.Series(losses).astype(float).plot()
    plt.show()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from load import config
from data_generator import DataGenerator
from model import net

import torch.optim as optim
import random
from tqdm import tqdm


class Trainer:

    def __init__(self, batches=1000, lr=0.01):
        self.optimizer = optim.SGD(net.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.batch_size = 10

        self.net = net
        if config['cuda']:
            self.net = self.net.cuda()

        self.n_data, self.n_lbl = DataGenerator.load()
        self.FILT = list(range(len(self.n_lbl)))

        self.batches = batches

    # @profile
    def create_batch(self):
        sample = random.sample(self.FILT, self.batch_size)
        t = self.n_data[sample]
        t = np.array(list(zip(*t))[1]) / 255
        # in your training loop:
        X = torch.Tensor(t.reshape(self.batch_size, 1, self.n_data[0][1].shape[0], -1))
        L = torch.Tensor(self.n_lbl[sample])
        if config['cuda']:
            X, L = X.cuda(), L.cuda()
        X, L = torch.autograd.Variable(X, requires_grad=True), torch.autograd.Variable(L, requires_grad=False)
        return X, L

    # @profile
    def train(self):
        losses = []
        for _ in tqdm(range(self.batches)):
            self.optimizer.zero_grad()

            X, L = self.create_batch()
            Y = self.net(X)

            loss = self.criterion(Y, L)
            l = loss.cpu() if config['cuda'] else loss
            losses.append(l.data.numpy())

            loss.backward()
            self.optimizer.step()
        return losses


if __name__ == '__main__':
    losses = Trainer(100).train()
    pd.Series(losses).astype(float).plot()
    plt.show()

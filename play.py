import numpy as np
import cv2

import torch

from load import config
from data_generator import DataGenerator
from model import net



class Player:

    def __init__(self, batches=1000, lr=0.01):
        self.net = net
        if config['cuda']:
            self.net = self.net.cuda()

        self.net.load()

    def play(self):
        cap = cv2.VideoCapture(0)

        scrx = config['screen_resolution']['x']
        scry = config['screen_resolution']['y']

        while True:
            board = np.zeros((scry, scrx))
            ret, frame = cap.read()
            dat = DataGenerator.extract(frame)
            if len(dat) < 2:
                continue
            X = torch.Tensor(np.array([[dat[1]]]) / 255).float()
            if config['cuda']:
                X = X.cuda()

            Y = self.net(torch.autograd.Variable(X))
            if config['cuda']:
                Y = Y.cpu()

            x, y = Y.data.numpy()[0]
            x, y = x * scrx, y * scry
            x, y = int(x), int(y)
            board = cv2.rectangle(board, (x - 20, y - 20), (x + 20, y + 20), 150, int(20 / 2))
            cv2.imshow('frame', board)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Player().play()

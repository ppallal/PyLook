import os

from load import config, face_cascade, eye_cascade
import cv2
import numpy as np
import random
import time
import pickle
from zipfile import ZipFile

import win32gui
import win32api
import glob
import sys

scrx = config['screen_resolution']['x']
scry = config['screen_resolution']['y']

frame_set_size = 100


class DataGenerator:

    ball_radius = 10
    velocity = 10

    frame_x_res = 100
    frame_y_res = 75

    frame_save_res = 32

    eye_resolution_multiplier = 3
    face_res = 64
    eye_res = face_res * eye_resolution_multiplier

    def __init__(self):
        self.data, self.labels = self.generate()
        self.n_data = np.array(self.data)
        self.n_lbl = np.array(self.labels)

    def save(self):
        if os.path.exists('data/n_data.pickle'):

            with open('data/n_data.pickle', 'rb') as f:
                n_data = pickle.load(f)
                n_data = np.hstack([n_data, self.n_data])

            with open('data/n_lbl.pickle', 'rb') as f:
                n_lbl = pickle.load(f)
                n_lbl = np.vstack([n_lbl, self.n_lbl])

        else:
            n_data = self.n_data
            n_lbl = self.n_lbl

        with open('data/n_data.pickle', 'wb') as f:
            pickle.dump(n_data, f)

        with open('data/n_lbl.pickle', 'wb') as f:
            pickle.dump(n_lbl, f)

    @staticmethod
    def load():
        with open('data/n_data.pickle', 'rb') as f:
            n_data = pickle.load(f)

        with open('data/n_lbl.pickle', 'rb') as f:
            n_lbl = pickle.load(f)

        return n_data, n_lbl

    @staticmethod
    def extract(frame, label=""):
        frame_x_res = DataGenerator.frame_x_res
        frame_y_res = DataGenerator.frame_y_res
        eye_resolution_multiplier = DataGenerator.eye_resolution_multiplier

        gray_scheme = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray_scheme, (frame_x_res, frame_y_res))
        gray_big = cv2.resize(gray_scheme, (eye_resolution_multiplier * frame_x_res, eye_resolution_multiplier * frame_y_res))
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 1:
            return []
        rvals = []
        full = cv2.resize(gray_scheme, (DataGenerator.frame_save_res, DataGenerator.frame_save_res))
        rvals.append(full)

        for (x, y, w, h) in faces:
            roi_gray = gray_big[eye_resolution_multiplier * y:eye_resolution_multiplier * (y + h), eye_resolution_multiplier * x:eye_resolution_multiplier * (x + w)]
            fac = gray[y:(y + h), x:(x + w)]
            fac = cv2.resize(fac, (DataGenerator.face_res, DataGenerator.face_res))
            rvals.append(fac)
            if label:
                cv2.imwrite("data/" + label + "_face.jpg", fac)

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                eye = roi_gray[ey:ey + eh, ex:ex + ew]
                eye = cv2.resize(eye, (DataGenerator.eye_res, DataGenerator.eye_res))
                if label:
                    cv2.imwrite("data/" + label + "_eye_%d.jpg" % i, eye)
                rvals.append(eye)

        return rvals

    @staticmethod
    def get_ball(x, y):
        br = DataGenerator.ball_radius
        img = np.zeros((scry, scrx))
        cv2.rectangle(img, (x - br, y - br), (x + br, y + br), 255, int(br / 2))
        return img

    @staticmethod
    def momentum_gen():
        br = DataGenerator.ball_radius
        x = random.randint(br, scrx - br)
        y = random.randint(br, scry - br)
        dirs = []
        for _ in range(frame_set_size):
            di = 2 * np.pi * random.random()
            dirs.append(di)
            dirs = dirs[-4:]
            di = np.mean(dirs)

            x = x + DataGenerator.velocity * np.cos(di)
            y = y + DataGenerator.velocity * np.sin(di)

            x = int(x % (scrx))
            y = int(y % (scry))
            yield (x, y)

    @staticmethod
    def frame_gen():
        motion = DataGenerator.momentum_gen()
        for x, y in motion:
            yield (DataGenerator.get_ball(x, y)), x, y

    @staticmethod
    def generate():
        cap = cv2.VideoCapture(0)

        b = False
        lbls = 1

        firsttime = True

        labels = []
        data = []
        font = cv2.FONT_HERSHEY_SIMPLEX

        while not b:
            # Capture frame-by-frame
            for i, (board, x, y) in enumerate(DataGenerator.frame_gen()):
                cv2.putText(board, "%d" % len(data), (int(scrx/2), int(scry/2)), font, 4, (100, 100, 100), 2, cv2.LINE_AA)
                cv2.imshow('frame', board)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    b = True
                    break
                if firsttime:
                    time.sleep(3)
                    firsttime = False

                time.sleep(1 / 60)
                ret, frame = cap.read()
                time.sleep(1 / 60)

                dat = DataGenerator.extract(frame)

                if len(dat) < 3:
                    continue
                data.append(dat)
                labels.append((x / scrx, y / scry))
                lbls += 1

        cap.release()
        cv2.destroyAllWindows()

        return data, labels


class DataGenerator2(DataGenerator):

    def __init__(self):
        generate = self.generate()
        if not os.path.exists('data'):
            os.mkdir('data')

        for d, t in generate:
            with open('data/instance_%f.pickle' % t, 'wb') as f:
                pickle.dump(d, f)

    @staticmethod
    def load():
        files = glob.glob("data/instance_*.pickle")
        data = []
        lbls = []
        for fn in files:
            with open(fn, 'rb') as f:
                dat = pickle.load(f)
                data.append(dat['data'])
                lbls.append(dat['lbl'])
        n_data = np.array(data)
        n_lbls = np.array(lbls)
        return n_data, n_lbls

    @staticmethod
    def generate():
        cap = cv2.VideoCapture(0)
        state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
        while True:
            a = win32api.GetKeyState(0x01)
            if a != state_left:  # Button state changed
                state_left = a
                flags, hcursor, (x, y) = win32gui.GetCursorInfo()
                dat, lbl, t = DataGenerator2.onclick(x, y, cap)
                if len(dat) < 3:
                    continue
                yield {"data": dat, "lbl": lbl}, t
            time.sleep(0.001)

    @staticmethod
    def onclick(x, y, cap):
        x, y = x / scrx, y / scry
        t = time.time()
        ret, frame = cap.read()
        dat = DataGenerator2.extract(frame)
        return dat, (x, y), t

    @staticmethod
    def delete():
        files = glob.glob("data/instance_*.pickle")
        for fn in files:
            os.remove(fn)

    @staticmethod
    def package():
        try:
            files = glob.glob('data/data_*.zip')
            cnt = int(sorted(files)[-1].split("_")[1].split('.')[0])
            cnt += 1
        except:
            cnt = 1
        files = glob.glob("data/instance_*.pickle")
        with ZipFile('data/data_%d.zip' % cnt, 'w') as myzip:
            for fn in files:
                myzip.write(fn)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'package':
        DataGenerator2.package()
    else:
        DataGenerator2()

import cv2
import json

config = json.loads(open('config.json').read())

if config['cuda']:
    face_cascade = cv2.CascadeClassifier('resources/opencv_haar/cuda/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('resources/opencv_haar/cuda/haarcascade_eye.xml')
else:
    face_cascade = cv2.CascadeClassifier('resources/opencv_haar/normal/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('resources/opencv_haar/normal/haarcascade_eye.xml')


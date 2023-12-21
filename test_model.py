import cv2
import time
import numpy as np
from grab_screen import grab_screen
from scancodes import PressKey, ReleaseKey, W, A, S, D
from get_keys import key_check
from alexnet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'gtav-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)

def goStraight():
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(W)

def goLeft():
    ReleaseKey(D)
    PressKey(A)
    PressKey(W)

def goRight():
    ReleaseKey(A)
    PressKey(D)
    PressKey(W)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            image = grab_screen(region=(0, 40, 800, 640))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (80, 60))

            prediction = model.predict([image.reshape(WIDTH, HEIGHT, 1)])[0]
            moves = list(np.around(prediction))
            print(moves, prediction)

            if moves == [1, 0, 0]:
                goLeft()
            elif moves == [0, 1, 0]:
                goStraight()
            elif moves == [0, 0, 1]:
                goRight()

            keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()
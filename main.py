import cv2
import time
import os
import numpy as np
from grab_screen import grab_screen
from scancodes import PressKey, ReleaseKey, W, A, S, D
from get_keys import key_check

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist')
    training_data = []

def keys_to_output(keys):
    #        [A, W, D]
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1

    return output

def main():
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)

    while True:
        image = grab_screen(region=(0, 40, 800, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (80, 60))

        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([image, output])

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)

main()
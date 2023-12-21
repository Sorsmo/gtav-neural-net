import cv2
import time
import numpy as np
from PIL import Image, ImageGrab
from scancodes import PressKey, ReleaseKey, W, A, S, D

def regionOfInterest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    
    return masked

while True:
    last_time = time.time()

    img = ImageGrab.grab(bbox=(0, 0, 800, 500))
    # cv2.imshow("original", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.Canny(gray, 200, 200)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = regionOfInterest(gray, np.array([[(10, 500), (10, 200), (300, 200), (500, 200), (800, 200), (800, 500)]]))
    cv2.circle(gray, (400, 400), 100, (0, 0, 0), -1)
    cv2.imshow("gray", gray)

    rho = 1
    theta = np.pi/180
    threshold = 100
    min_line_length = 20
    max_line_gap = 15
    line_image = np.copy(img) 
    lines = cv2.HoughLinesP(gray, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(y1 - y2) < 100:
                    continue
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
    
    cv2.imshow("lines", cv2.cvtColor(np.array(line_image), cv2.COLOR_RGB2BGR))

    print("Loop time: ", time.time() - last_time)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
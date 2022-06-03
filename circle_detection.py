import cv2
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import *
import time
import math

yellowLower=(29,127,121)
yellowHigher=(179,255,255)

def isclose(a,b,maxi):
    if a <= b+maxi and a>=b-maxi:
        return True
    else:
        return False

cap = cv2.VideoCapture("data/video2.webm")
circles = []

#inimg = cv2.imread("data/salle2.jpg")

#h = int(inimg.shape[0]/2)
#w = int(inimg.shape[1]/2)
#dim = (w,h)

#inimg = cv2.resize(inimg,dim,interpolation=cv2.INTER_AREA)

prevCenter = (0,0)

while True:

    ret, inimg = cap.read()
    if not ret:
        print("error reading video")
        break

    ############################ Circles detection

    # Convert to grayscale.
    gray = cv2.cvtColor(inimg, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40)

    # only proceed if at least one circle was found
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        # store the circles in an array for later
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(inimg, (a, b), r, (0, 255, 0), 2)
    
    cv2.imshow("circles",inimg)
    #time.sleep(0.03)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
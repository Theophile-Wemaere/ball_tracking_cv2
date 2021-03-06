import cv2
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import *
import time
import math

# yellowLower = (29,127,121)
# yellowHigher = (179,255,255)

yellowLower, yellowHigher = (24, 105, 38), (51, 255, 255)


def isclose(a,b,maxi):
    if a <= b+maxi and a>=b-maxi:
        return True
    else:
        return False

def nothing(x):
    pass

cap = cv2.VideoCapture("data/video3.mp4")

cv2.namedWindow("output")
cv2.createTrackbar("Delay", "output", 0, 100,nothing)
#inimg = cv2.imread("data/salle2.jpg")

#h = int(inimg.shape[0]/2)
#w = int(inimg.shape[1]/2)
#dim = (w,h)

#inimg = cv2.resize(inimg,dim,interpolation=cv2.INTER_AREA)

prevCenter = (0,0)
x,y=0,0

while True:

    ret, inimg = cap.read()
    if not ret:
        print("error reading video")
        break

    circle_img = np.copy(inimg)
    mask_img = np.copy(inimg)

    ######################################################## Mask detection

    blurred = cv2.GaussianBlur(inimg,(11,11),0)
    hsv = cv2.cvtColor(inimg,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,yellowLower,yellowHigher)
    mask = cv2.erode(mask,None,iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            dist = math.sqrt((center[0]-prevCenter[0])**2+(center[1]-prevCenter[1])**2)
            #print("center : " + str(center) + " | prevCenter : " + str(prevCenter) + " | distance : " + str(dist))
            prevCenter=center
        except ZeroDivisionError:
            print("error, division by zero")

        if radius > 5 and radius < 140 : # min 5 ?
            print(radius)
            cv2.circle(mask_img, (int(x), int(y)), int(radius),(0, 0, 255,255), 2)

        # for pt in circles:
        #     if isclose(x, pt[0], 5) and isclose(y,pt[1],5) and isclose(radius, pt[2], 5):
        #         cv2.circle(inimg, (int(x), int(y)), int(radius),(0, 255, 255,255), 2)
        #         cv2.circle(inimg, (pt[0], pt[1]), pt[2], (0, 255, 0), 2)
        #         break
        #         #print("mask center = ({},{}) and radius = {} | circle center = ({},{}) and radius = {}".format(int(x),int(y),int(radius),pt[0],pt[1],pt[2]))
                
    ######################################################## Circles detection

    gray = cv2.cvtColor(inimg, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40)
    #circles = []

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        # store the circles in an array for later
        new_frame = 1
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            #circles.append((a,b,r))
            cv2.circle(circle_img, (a, b), r, (0, 255, 0), 2)

            if isclose(x, a, 5) and isclose(y,b,5) and isclose(radius, r, 1) and new_frame == 1:
                cv2.circle(inimg, (int(x), int(y)), int(radius),(0, 255, 255,255), 2)
                cv2.circle(inimg, (a, b), r, (0, 255, 0), 2)
                new_frame = 0
 
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    gray_blurred = cv2.cvtColor(gray_blurred, cv2.COLOR_GRAY2BGR)
    stacked1 = np.hstack((mask,mask_img))
    stacked2 = np.hstack((circle_img,inimg))
    stacked = np.vstack((stacked1,stacked2))
    cv2.imshow("output",cv2.resize(stacked,None,fx=0.8,fy=0.8))

    delay = cv2.getTrackbarPos("Delay", "output")
    time.sleep(delay/100)

    # type q to leave the video or space to play/pause the video 

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == 32:
        cv2.waitKey()

cap.release()
cv2.destroyAllWindows()
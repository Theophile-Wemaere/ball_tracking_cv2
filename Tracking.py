import cv2
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import *
import time
import math

yellowLower=(29,127,121)
yellowHigher=(179,255,255)

buffer_size=0
pts = deque(maxlen=buffer_size)

def showIMG(name,image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testRGB(image):
    blurred = cv2.GaussianBlur(image,(11,11),0)
    hsv = cv2.cvtColor(blurred,cv2.COLOR_RGB2HSV)

    root1=Tk()
    root2=Tk()
    lower=RGB_selector(root1,"mininmum")
    higher=RGB_selector(root2,"maximum")

    while True:
        #
        b_m,g_m,r_m=lower.b.get(),lower.g.get(),lower.r.get()
        b_p,g_p,r_p=higher.b.get(),higher.g.get(),higher.r.get()

        moins=(b_m,g_m,r_m)
        plus=(b_p,g_p,r_p)

        print(moins,plus)

        mask = cv2.inRange(hsv,moins,plus)
        mask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow("Frame",mask)

        lower.page.update()
        higher.page.update()

        lower.button.config(bg=_from_rgb((lower.r.get(),lower.g.get(),lower.b.get())))
        higher.button.config(bg=_from_rgb((higher.r.get(),higher.g.get(),higher.b.get())))

        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    lower.page.destroy()
    higher.page.destroy()

def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'

class RGB_selector(tk.Frame):
    def __init__(self, page, titre):
        self.page=page
        self.titre=titre
        self.page.title(self.titre)
        self.page.geometry("210x310")
        self.b = Scale(page, from_=0, to=255, orient=HORIZONTAL, length=200)
        self.b.pack()
        self.g = Scale(page, from_=0, to=255, orient=HORIZONTAL, length=200)
        self.g.pack()
        self.r = Scale(page, from_=0, to=255, orient=HORIZONTAL, length=200)
        self.r.pack()

        self.button=Button(page, text="",width=20,height=10)


        self.button.pack(pady=20)


cap = cv2.VideoCapture("data/video2.webm")

#inimg = cv2.imread("data/salle2.jpg")

#h = int(inimg.shape[0]/2)
#w = int(inimg.shape[1]/2)
#dim = (w,h)

#inimg = cv2.resize(inimg,dim,interpolation=cv2.INTER_AREA)

#testRGB(inimg)

prevCenter = (0,0)

while True:

    ret, inimg = cap.read()
    if not ret:
        print("error reading video")
        break

    blurred = cv2.GaussianBlur(inimg,(11,11),0)
    hsv = cv2.cvtColor(inimg,cv2.COLOR_BGR2HSV)

    #apply mask
    mask = cv2.inRange(hsv,yellowLower,yellowHigher)
    mask = cv2.erode(mask,None,iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            dist = math.sqrt((center[0]-prevCenter[0])**2+(center[1]-prevCenter[1])**2)
            print("center : " + str(center) + " | prevCenter : " + str(prevCenter) + " | distance : " + str(dist))
            prevCenter=center
        except ZeroDivisionError:
            print("error, division by zero")
        # only proceed if the radius meets a minimum size and the center isn't too far for the previous one (no tp)
        if radius > 1 and dist < 50:
            # print(dist)
            # draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(inimg, (int(x), int(y)), int(radius),(0, 255, 255,255), 2)
            cv2.circle(inimg, center, 5, (0, 0, 255,255), -1)
    
    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
        cv2.line(inimg, pts[i - 1], pts[i], (0,0, 255, 255), thickness)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((mask,inimg))
    cv2.imshow("circles",cv2.resize(stacked,None,fx=0.8,fy=0.8))
    time.sleep(0.03)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#finding hsv range of target object(pen)
import cv2
import numpy as np
import time
# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

img_l = [".jpg",".jpeg",".png"]
ftype = "vid"

file="data/test2_vid3.png"

for i in img_l:
    if file.find(i) != -1:
        ftype = "img"
    
if ftype == "vid":
    # Initializing the webcam feed.
    cap = cv2.VideoCapture(file)
    #cap.set(3,1280)
    #cap.set(4,720)
else:
    frame = cv2.imread(file)    
    #h = int(frame.shape[0]²/2)
    #w = int(frame.shape[1]/2)
    #dim = (w,h)
    #frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of
# H,S and V channels. The Arguments are like this: Name of trackbar,
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", 29, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 127, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 121, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Delay", "Trackbars", 0, 1000, nothing)

while True:

    if ftype == "vid":
        # Start reading the webcam feed frame by frame.
        ret, frame = cap.read()
        if not ret:
            break   

    # Flip the frame horizontally (Not required)
    #frame = cv2.flip( frame, 1 )

    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the new values of the trackbar in real time as the user changes
    # them
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Converting the binary mask to 3 channel image, this is just so
    # we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3,frame,res))

    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))

    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # If the user presses `s` then print this array.
    if key == ord('s'):

        thearray = [(l_h,l_s,l_v),(u_h, u_s, u_v)]
        print(thearray)
        name = input("choose a name to save the array : ")
        f=open(name,"w")
        f.write(str(thearray))
        f.close()

        # Also save this array as penval.npy
        #np.save('hsv_value',thearray)
        #break

    delay = cv2.getTrackbarPos("Delay", "Trackbars")
    time.sleep(delay/1000)

# Release the camera & destroy the windows.
#cap.release()
cv2.destroyAllWindows()

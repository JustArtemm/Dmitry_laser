import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
    # recolored = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imwrite( 'data/image.png',frame)
    # Display the resulting frame
        cv2.imshow('image', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
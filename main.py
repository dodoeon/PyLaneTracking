from http.client import CannotSendHeader
import cv2
from cv2 import bilateralFilter
import numpy as np

Drive = cv2.VideoCapture('drive.mp4')
roi = cv2.imread('roi.png')

while Drive.isOpened():
    run, frame = Drive.read()
    if not run:
        print("File not found")
        break
    RGB = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = (13, 10, 195)
    upper = (27, 45, 255)

    mask = cv2.bitwise_and(HSV, roi)
    mask = cv2.inRange(mask, lower, upper)

    mask = bilateralFilter(mask, -1, 10, 5)
    mask = cv2.Canny(mask, 100, 200)

    RGB = cv2.resize(RGB, (640, 360))
    mask = cv2.resize(mask, (640, 360))
    cv2.imshow('mask', mask)
    cv2.imshow('video', RGB)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


Drive.release()
cv2.destroyAllWindows()

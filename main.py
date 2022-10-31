
import cv2

Drive = cv2.VideoCapture('drive.mp4')
roi = cv2.imread('roi_720.png')

while Drive.isOpened():
    run, frame = Drive.read()
    if not run:
        print("File not found")
        break
    RGB = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = (0, 0, 190)
    upper = (120, 30, 255)

    mask = cv2.bilateralFilter(HSV, 2, 10, 10)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.inRange(mask, lower, upper)

    mask = cv2.Canny(mask, 200, 200)

    RGB = cv2.resize(RGB, (960, 540))
    mask = cv2.resize(mask, (960, 540))
    cv2.imshow('mask', mask)
    cv2.imshow('video', RGB)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


Drive.release()
cv2.destroyAllWindows()

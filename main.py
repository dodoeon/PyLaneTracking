
import cv2
import numpy as np

Drive = cv2.VideoCapture('drive.mp4')
roi = cv2.imread('roi_1920.png')


def lane_detection_HSV(Frame):
    lower = (0, 5, 195)
    upper = (180, 35, 255)
    mask = cv2.bilateralFilter(Frame, 2, 10, 10)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.inRange(mask, lower, upper)
    return mask


def lane_detection_RGB(Frame):
    lower = (190, 190, 190)
    upper = (255, 255, 255)
    mask = cv2.bilateralFilter(Frame, 2, 10, 10)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.inRange(mask, lower, upper)
    return mask


def draw_lines(img, lines, color=[0, 255, 0], thickness=4):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img):
    return cv2.addWeighted(initial_img, 1, img, 1, 0)


while Drive.isOpened():
    run, frame = Drive.read()
    if not run:
        print("File not found")
        break
    RGB = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    HSV_mask = lane_detection_HSV(HSV)
    RGB_mask = lane_detection_RGB(RGB)

    MIX_mask = cv2.bitwise_or(HSV_mask, RGB_mask)
    MIX_mask = cv2.Canny(MIX_mask, 70, 200)

    MIX_mask = hough_lines(MIX_mask, 1, 1 * np.pi/180, 30, 90, 250)  # 허프 변환

    Result = weighted_img(MIX_mask, RGB)
    Result = cv2.resize(Result, (960, 540))

    cv2.imshow('video', Result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


Drive.release()
cv2.destroyAllWindows()

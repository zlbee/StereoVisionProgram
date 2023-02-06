# Zhangli WANG, 20028336, scyzw1@nottingham.edu.cn

import cv2

trackbarWindowName = 'Disparity Map with Parameter Tunning'


def nothing(x):
    pass


def draw_trackbar():
    cv2.namedWindow(trackbarWindowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(trackbarWindowName, 800, 900)
    cv2.createTrackbar('windowSize', trackbarWindowName, 1, 15, nothing)
    cv2.createTrackbar('numDisparities', trackbarWindowName, 1, 20, nothing)
    cv2.createTrackbar('blockSize', trackbarWindowName, 1, 50, nothing)
    return trackbarWindowName


def get_trackbar_values():
    # Updating the parameters based on the trackbar positions
    windowSize = cv2.getTrackbarPos('windowSize', trackbarWindowName)
    numDisparities = cv2.getTrackbarPos('numDisparities', trackbarWindowName) * 16
    blockSize = cv2.getTrackbarPos('blockSize', trackbarWindowName) * 2 + 5
    return numDisparities, blockSize, windowSize
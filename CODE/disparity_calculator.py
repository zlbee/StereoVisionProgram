# Zhangli WANG, 20028336, scyzw1@nottingham.edu.cn

import argparse

import cv2
import numpy as np
import trackbar_controller

# https://www.programcreek.com/python/example/110664/cv2.StereoSGBM_create
def calc_disp_w_filter(leftImg, rightImg):
    # parameters
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=10 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=1,
        P1=8 * 3 * window_size,
        P2=32 * 3 * window_size,
        disp12MaxDiff=-1,
        uniquenessRatio=15,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    ld = 80000
    sigma = 1.5

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(ld)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(leftImg, rightImg)  # .astype(np.float32)/16
    dispr = right_matcher.compute(rightImg, leftImg)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, leftImg, None, dispr)  # important to put "imgL" here!!!

    # normalize
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    filteredImg = np.uint8(filteredImg)

    return filteredImg


def calc_disp(left_img, right_img):
    window_size = 3

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=10 * 16,  # black bar
        blockSize=1,
        P1=8 * 3 * window_size,
        P2=32 * 3 * window_size,
        disp12MaxDiff=-1,
        uniquenessRatio=15,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_img, right_img)

    # normalize
    disparity = cv2.normalize(src=disparity, dst=disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    disparity = np.uint8(disparity)

    return disparity


def calc_disp_by_pms(left_img, right_img, numDisparities, blockSize, window_size):
    # initialize stereo calculator
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=10 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=1,
        P1=8 * 3 * window_size,
        P2=32 * 3 * window_size,
        disp12MaxDiff=-1,
        uniquenessRatio=15,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    ld = 80000
    sigma = 1.5

    # set parameters
    left_matcher.setNumDisparities(numDisparities)
    left_matcher.setBlockSize(blockSize)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(ld)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(left_img, right_img)  # .astype(np.float32)/16
    dispr = right_matcher.compute(right_img, left_img)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, left_img, None, dispr)  # important to put "imgL" here!!!

    # # create stereo algorithm
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=10 * 16,  # black bar
    #     blockSize=1,
    #     disp12MaxDiff=-1,
    #     uniquenessRatio=15,
    #     speckleWindowSize=50,
    #     speckleRange=32,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )
    #
    # # set parameters
    # stereo.setNumDisparities(numDisparities)
    # stereo.setBlockSize(blockSize)
    #
    # # compute disparities
    # disparity = stereo.compute(left_img, right_img)

    # normalize
    disparity = cv2.normalize(src=disparity, dst=disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    disparity = np.uint8(disparity)

    return disparity


def dynamical_draw_disp(left_img, right_img):
    # draw trackbars
    windowName = trackbar_controller.draw_trackbar()
    while True:
        # get values from trackbars
        numDisparites, blockSize, windowSize = trackbar_controller.get_trackbar_values()

        if numDisparites == 0:
            numDisparites = 1
        # calculate disparity
        disparity = calc_disp_by_pms(left_img, right_img, numDisparites, blockSize, windowSize)

        # display disparity map
        cv2.imshow(windowName, disparity)

        # terminate condition
        if cv2.waitKey(2) == 27:
            cv2.destroyAllWindows()
            break
    return disparity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Disparity')
    parser.add_argument('--left_image', type=str, required=True, help='Left image.')
    parser.add_argument('--right_image', type=str, required=True, help='Right image.')
    args = parser.parse_args()
    
    calc_disp_w_filter(args.left_image, args.right_image)
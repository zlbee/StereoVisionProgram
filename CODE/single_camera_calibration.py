# reference: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

import argparse
import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt

def camera_calibrate(chessboards_path, save_name):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    width = 9
    height = 6
    squareLength = 0.025
    chessboard_size = (width, height)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * squareLength  # convert to real-world size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # load chessboard images
    # image_paths = glob.glob('./resources/chessboards/*.jpg')
    image_paths = glob.glob(chessboards_path)

    # iterate through images and find chessboard corners
    stacking_images = []
    for fileName in image_paths:
        # read image
        img = cv2.imread(fileName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size)

        # if find corners, add them to objpoints and imgpoints
        if ret:
            # add to objpoints
            objpoints.append(objp)

            # refine and add to imgpoints
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # draw the image with corners
            cornered_img = img.copy()
            cv2.drawChessboardCorners(cornered_img, chessboard_size, corners2, ret)

            # stacking images side-by-side
            stacking_image = np.hstack((img, cornered_img))
            stacking_images.append(stacking_image)
        else:
            print("Corners of " + fileName + " not found.")

    # display and save cornered images
    for i in range(len(stacking_images)):
        plt.figure('Camera Calibration Corners Locations %d' % (i + 1), figsize=(16, 9))
        # pyplot.subplot(1, 1, 1)
        plt.imshow(stacking_images[i], 'gray')
        plt.title('Camera Calibration Corners Locations')
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./resources/camera_calibration_results/' + save_name + '%02d.png' % (i + 1))
    # plt.show()

    # calibrate camera
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # print RMS error
    print("RMS: " + str(ret))

    # save results
    # cv_file = cv2.FileStorage('./camera_parameters.yml', cv2.FILE_STORAGE_WRITE)
    cv_file = cv2.FileStorage('./' + save_name, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", K)  # save camera intrinsic matrix
    cv_file.write("D", D)  # save distortion coefficients
    cv_file.release()

    print("finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('--chessboards_path', type=str, required=True, help='Chessboard images path.')
    parser.add_argument('--save_name', type=str, required=True, help='Save file name.')
    args = parser.parse_args()

    camera_calibrate(args.chessboards_path, args.save_name)
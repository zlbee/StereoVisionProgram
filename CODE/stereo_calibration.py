# reference: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

import cv2
import numpy as np
import glob
import single_camera_calibration


def stereo_cameras_calibrate():
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
    left_imgpoints = []  # 2d points in image plane.
    right_imgpoints = []  # 2d points in image plane.

    # load chessboard images
    left_img_paths = glob.glob('./resources/left_chessboards/*.jpg')
    right_img_paths = glob.glob('./resources/right_chessboards/*.jpg')

    # load cameras parameters
    left_file = 'left_camera_parameters.yml'
    right_file = 'right_camera_parameters.yml'
    single_camera_calibration.camera_calibrate('./resources/left_chessboards/*.jpg', left_file)
    single_camera_calibration.camera_calibrate('./resources/right_chessboards/*.jpg', right_file)
    cv_file = cv2.FileStorage('./' + left_file, cv2.FILE_STORAGE_READ)
    left_K = cv_file.getNode('K').mat()
    left_D = cv_file.getNode('D').mat()
    cv_file = cv2.FileStorage('./' + right_file, cv2.FILE_STORAGE_READ)
    right_K = cv_file.getNode('K').mat()
    right_D = cv_file.getNode('D').mat()
    cv_file.release()


    # sort images to match pairs
    left_img_paths.sort()
    right_img_paths.sort()
    img_paths_pairs = zip(left_img_paths, right_img_paths)

    # iterate through images and find chessboard corners
    for left_img_path, right_img_path in img_paths_pairs:
        # read images
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)

        # find chessboard corners
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, chessboard_size)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard_size)

        if left_ret and right_ret:
            # add to objpoints
            objpoints.append(objp)

            # refine and add to imgpoints
            left_corners2 = cv2.cornerSubPix(left_gray, left_corners, (5, 5), (-1, -1), criteria)
            right_corners2 = cv2.cornerSubPix(right_gray, right_corners, (5, 5), (-1, -1), criteria)
            left_imgpoints.append(left_corners2)
            right_imgpoints.append(right_corners2)
        else:
            print("Corners of " + left_img_path + " and " + right_img_path + "not found.")

    # calibrate cameras
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, left_K, left_D, right_K, right_D, left_gray.shape[::-1])

    # print RMS error
    print("Stereo cameras calibration RMS error: " + str(ret))

    # save results
    cv_file = cv2.FileStorage('./stereo_cameras_parameters.yml', cv2.FILE_STORAGE_WRITE)
    cv_file.write("K1", K1)  # save camera 1 intrinsic matrix
    cv_file.write("D1", D1)  # save camera 1 distortion coefficients
    cv_file.write("K2", K2)  # save camera 2 intrinsic matrix
    cv_file.write("D2", D2)  # save camera 2 distortion coefficients
    cv_file.write("R", R)  # save rotation matrix of first camera to second camera
    cv_file.write("T", T)  # save translation matrix of first camera to second camera
    cv_file.write("E", E)  # save essential matrix
    cv_file.write("F", F)  # save fundamental matrix
    cv_file.release()

    print("finished")


if __name__ == '__main__':
    stereo_cameras_calibrate()
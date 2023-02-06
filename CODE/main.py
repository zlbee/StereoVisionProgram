# Zhangli WANG, 20028336, scyzw1@nottingham.edu.cn
import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt
import disparity_calculator
import epilines_drawer
import depth_calculator
import trackbar_controller
import disparity_calculator_self_implement


def execute(algorithm_type, left_img_name, right_img_name):
    """
    :param algorithm_type: 'Types of algorithm:'
                                                           '1 SGBM Algorithm with tunning parameters on trackbars,
                                                           press ESC to quit'
                                                           '2 SGBM Algorithm'
                                                           '3 SGBM Algorithm with filter'
                                                           '4 My Algorithm with SAD'
                                                           '5 My Algorithm with SSD'
    :param left_img_name: input left image name: l_img_set_1.jpg, l_img_set_2.jpg, l_img_set_3.jpg
    :param right_img_name: input right image name: r_img_set_1.jpg, r_img_set_2.jpg, r_img_set_3.jpg
    :return: None
    """

    flag = algorithm_type
    if algorithm_type is None:
        flag = 3
    if left_img_name is None:
        left_img_name = 'l_img_set_1.jpg'
    if right_img_name is None:
        right_img_name = 'r_img_set_1.jpg'

    # read image
    left_img = cv2.imread('./resources/images/'+left_img_name)
    right_img = cv2.imread('./resources/images/'+right_img_name)

    img_size = tuple(reversed(left_img.shape[:2]))

    # 1.calibrate camera (having executed stereo_calibration.py so no operation here)

    # read camera parameters
    cv_file = cv2.FileStorage('./stereo_cameras_parameters.yml', cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode('K1').mat()
    D1 = cv_file.getNode('D1').mat()
    K2 = cv_file.getNode('K2').mat()
    D2 = cv_file.getNode('D2').mat()
    R = cv_file.getNode('R').mat()
    T = cv_file.getNode('T').mat()
    cv_file.release()

    # optional. undistort image
    optimal_K1, roi1 = cv2.getOptimalNewCameraMatrix(K1, D1, img_size, 1, img_size)
    optimal_K2, roi2 = cv2.getOptimalNewCameraMatrix(K2, D2, img_size, 1, img_size)
    undistort_left_img = cv2.undistort(left_img, K1, D1, optimal_K1, None)
    undistort_right_img = cv2.undistort(right_img, K2, D2, optimal_K2, None)

    # 2.rectify and undistort images
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    rectified_left_img = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)
    rectified_right_img = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

    # 3.compute disparity
    gray_left_img = cv2.cvtColor(rectified_left_img, cv2.COLOR_RGB2GRAY)
    gray_right_img = cv2.cvtColor(rectified_right_img, cv2.COLOR_RGB2GRAY)
    if flag == 1:
        disparity = disparity_calculator.dynamical_draw_disp(gray_left_img, gray_right_img)
    elif flag == 2:
        disparity = disparity_calculator.calc_disp(gray_left_img, gray_right_img)
    elif flag == 3:
        disparity = disparity_calculator.calc_disp_w_filter(gray_left_img, gray_right_img)
    else:
        if flag == 4:
            disparity = disparity_calculator_self_implement.calc_disp(gray_left_img, gray_right_img, flag=True,
                                                                      algorithm=1)  # SAD
        else:
            disparity = disparity_calculator_self_implement.calc_disp(gray_left_img, gray_right_img, flag=True,
                                                                      algorithm=2)  # SSD

    # 4.estimate depth
    depth = depth_calculator.calc_depth(disparity, K1)

    # optional. draw epilines
    epilines_drawer.draw_epilines(gray_left_img, gray_right_img)

    # plot and save images
    plt.subplot(3, 2, 1), plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    plt.title('Left'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 2, 2), plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    plt.title('Right'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 2, 3), plt.imshow(cv2.cvtColor(undistort_left_img, cv2.COLOR_BGR2RGB))
    plt.title('Undistorted Left'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 2, 4), plt.imshow(cv2.cvtColor(undistort_right_img, cv2.COLOR_BGR2RGB))
    plt.title('Undistorted Right'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 2, 5), plt.imshow(cv2.cvtColor(rectified_left_img, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Left'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 2, 6), plt.imshow(cv2.cvtColor(rectified_right_img, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Right'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.savefig('./resources/output/img_processes.png', dpi=500)
    plt.show()

    plt.subplot(1, 1, 1), plt.imshow(disparity, 'gray')
    plt.title('Disparity Map'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.gcf().set_size_inches(img_size[0] / 100, img_size[0] / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('./resources/output/disparity.png', dpi=500)
    plt.show()
    plt.subplot(1, 1, 1), plt.imshow(depth, 'gray')
    plt.title('Depth Map'), plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.gcf().set_size_inches(img_size[0] / 100, img_size[0] / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('./resources/output/depth.png', dpi=500)
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--algorithm_type', type=int, help='Types of algorithm:'
                                                           '1 SGBM Algorithm with tunning parameters on trackbars'
                                                           '2 SGBM Algorithm'
                                                           '3 SGBM Algorithm with filter'
                                                           '4 My Algorithm with SAD'
                                                           '5 My Algorithm with SSD')
    parser.add_argument('--left_img', type=str, help='input left image name')
    parser.add_argument('--right_img', type=str, help='input right image name')
    args = parser.parse_args()

    execute(algorithm_type=args.algorithm_type, left_img_name=args.left_img, right_img_name=args.right_img)
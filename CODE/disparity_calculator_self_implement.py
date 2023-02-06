# Zhangli WANG, 20028336, scyzw1@nottingham.edu.cn

import argparse
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags
from skimage.draw import line


def calc_disp(left_img, right_img, flag, algorithm):
    print("running self implemented disparity calculator, please wait for about 3 minutes.")
    # read camera parameters
    cv_file = cv2.FileStorage('./stereo_cameras_parameters.yml', cv2.FILE_STORAGE_READ)
    F = cv_file.getNode('F').mat()
    cv_file.release()

    # parameters
    frame_offset = 6               # offset of the frame
    patch_size = 3                  # actually it is half of patch length
    height, width = left_img.shape
    stride = 3                      # stride of window sliding in the right image
    shrink_rate = 4                 # rate of shrinking raw data to reduce computational expenses
    search_scale = 18                # search scale coefficient for epiline/scanline

    # shrink image
    left_img = cv2.resize(left_img, (int(width/shrink_rate), int(height/shrink_rate)), interpolation=cv2.INTER_AREA)
    right_img = cv2.resize(right_img, (int(width/shrink_rate), int(height/shrink_rate)), interpolation=cv2.INTER_AREA)
    height = int(height/shrink_rate)
    width = int(width/shrink_rate)

    # initialize disparity map
    disp_map = np.zeros((height, width), dtype=np.uint8)

    # iterate all pixels of left image
    right_ep_img = right_img.copy()
    for h in range(frame_offset, height-frame_offset-1):
        for w in range(frame_offset, width-frame_offset-1):
            # 1. find the left point
            left_p = np.array([h, w]).reshape(-1, 1, 2)
            y = left_p[0][0][0]

            # unrectified case
            if flag == False:
                # 2. if unrectified, find all points in the epiline
                epilinesR = cv2.computeCorrespondEpilines(left_p, 2, F)
                drawLines(right_ep_img, epilinesR)
                x1 = frame_offset
                if w > frame_offset + patch_size * search_scale:
                    x1 = w - patch_size * search_scale
                y1 = -(epilinesR[0][0][2]+epilinesR[0][0][0]*x1)/epilinesR[0][0][1]
                x2 = width-frame_offset
                if width - w > frame_offset + patch_size * search_scale:
                    x2 = w + patch_size * search_scale
                y2 = -(epilinesR[0][0][2]+epilinesR[0][0][0]*x2)/epilinesR[0][0][1]
                points_epilineR = np.linspace((x1, y1), (x2, y2), (int)(width/stride)).astype(int)
                # print(y)

            # rectified case
            else:
                # 2. if rectified, find all points in the scanline
                x1 = frame_offset
                if w > frame_offset + patch_size * search_scale:
                    x1 = w - patch_size * search_scale
                x2 = width-frame_offset
                if width - w > frame_offset + patch_size * search_scale:
                    x2 = w + patch_size * search_scale
                points_epilineR = np.array([[x, y] for x in range(x1, x2, stride)])
                # print(y)

            # 3. find the patch of the left image
            left_patch = left_img[h-patch_size:h+patch_size, w-patch_size:w+patch_size]

            # 4. find the patch of the right image along with the epiline
            p_SAD_list = []
            for p in points_epilineR:
                right_patch = right_img[p[1]-patch_size:p[1]+patch_size, p[0]-patch_size:p[0]+patch_size]
                # 5. calculate SAD/SSD
                cost = 0
                if algorithm == 1:
                    for i in range(left_patch[0].size):
                        for j in range(left_patch[1].size):
                            diff = abs(left_patch[i][j] - right_patch[i][j])
                            cost += diff
                else:
                    for i in range(left_patch[0].size):
                        for j in range(left_patch[1].size):
                            diff = (left_patch[i][j] - right_patch[i][j]) ** 2
                            cost += diff
                p_SAD_list.append((p, cost))

            # 6. find the best match point in the epiline
            min_p_cost = min(p_SAD_list, key=lambda e: e[1])

            # 7. calculate point disparity
            disp = np.absolute(min_p_cost[0][0]-w)  # horizontal distance
            disp_map[h, w] = disp

    # normalize
    disp_map = cv2.normalize(src=disp_map, dst=disp_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    disp_map = np.uint8(disp_map)

    print("disp calculation self implemented finished")

    return disp_map


# https://stackoverflow.com/questions/51089781/how-to-calculate-an-epipolar-line-with-a-stereo-pair-of-images-in-python-opencv
# draw the provided lines on the image
def drawLines(img, lines):
    _, c = img.shape
    for r in lines:
        x0, y0 = map(int, [0, -r[0][2]/r[0][1]])
        x1, y1 = map(int, [c, -(r[0][2]+r[0][0]*c)/r[0][1]])
        cv2.line(img, (x0, y0), (x1, y1), [0, 0, 255], 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Disparity')
    parser.add_argument('--left_image', type=str, required=True, help='Left image.')
    parser.add_argument('--right_image', type=str, required=True, help='Right image.')
    args = parser.parse_args()

    calc_disp(args.left_image, args.right_image, True, 2)

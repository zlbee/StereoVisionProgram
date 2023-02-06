# reference: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

# calculate depth
import cv2
import numpy as np

def calc_depth(disparity, K):
    # acquire intrinsic coefficients
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    baseline = 7.9  # measured as a hyper parameter

    # initialize depth map
    img_size = tuple((disparity.shape[:2]))
    depth = np.zeros(img_size, dtype=np.uint8)

    # iterate to generate depth map
    for w in range(img_size[0]):
        for h in range(img_size[1]):
            if disparity[w, h] == 0:
                continue
            depth[w, h] = fx * baseline / disparity[w, h]

    # normalize
    depth = cv2.normalize(src=depth, dst=depth, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    print("depth calculation finished")
    return depth




# README
camera_calibration_results: directory of saving drawn corners of calibration.
images: directory of input images
left_chessboard right_chessboard: directory of chessboard images for calibration.
output: output directory.

scyzw1.yaml: my anaconda environment.

depth_calculator.py: calculate depth by disparity
disparity_calculator.py: SGBM algorithms
disparity_calculator_self_implement.py: my disparity algorithms
epilines_drawer.py: draw epipolar lines with SIFT
main.py: entry point
single_camera_calibration.py: calibrate single camera
stereo_calibration.py: calibrate stereo camera
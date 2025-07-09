import cv2
import numpy as np
import glob

pattern_size = (9, 6)
square_size = 0.015  # in meters (adjust to real square size)

# Prepare object points like (0,0,0), (1,0,0), ..., in 3D space
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D world points
imgpoints = []  # 2D image points
gray_shape = None

images = glob.glob('calibration_images/*.png')
for fname in images:
    global gray
    gray = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
        objpoints.append(objp)
        imgpoints.append(corners)
        if gray_shape is None:
            gray_shape = gray.shape[::-1]

ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None
)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Save for later use
np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
import cv2
import cv2.aruco as aruco
import numpy as np
import glob

# Set Charuco board parameters (must match your printed board)
squares_x = 5         # number of squares along the X direction
squares_y = 7         # number of squares along the Y direction
square_length = 0.04  # meters
marker_length = 0.02  # meters

# Load ArUco dictionary and create Charuco board
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
charuco_board = aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)

# Prepare lists to collect calibration data
all_charuco_corners = []
all_charuco_ids = []
image_size = None

# Load images
images = glob.glob('charuco_images/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None and len(ids) > 0:
        # Interpolate Charuco corners
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )

        if retval > 0:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)

# Perform camera calibration (ret stands for )
ret, camera_matrix, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=charuco_board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Save calibration data
np.savez("charuco_calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
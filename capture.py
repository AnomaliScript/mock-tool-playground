import cv2
import cv2.aruco as aruco
import os

# Trial number for .png organization
trial_number = input("Which trial is this? (int): ")

# Define CharUco board parameters
squares_x = 5  # number of squares in X direction
squares_y = 7  # number of squares in Y direction
square_length = 0.04  # in meters (used later for calibration)
marker_length = 0.02  # in meters (used later for calibration)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50) # Common options: DICT_4X4_50 DICT_5X5_100 DICT_6X6_250 DICT_7X7_1000
charuco_board = aruco.CharucoBoard_create(
    squares_x,           # Number of squares along X (columns)
    squares_y,           # Number of squares along Y (rows)
    square_length,       # Size of a square (in meters or any consistent unit)
    marker_length,       # Size of each ArUco marker inside the squares
    aruco_dict           # The ArUco dictionary from Dictionary_get()
)

# Save folder
save_folder = "charuco_images"

# Start camera
cap = cv2.VideoCapture(0)
img_id = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(
        grayed_frame,     # Image
        aruco_dict,       # Dictonary that we're using
        parameters=None   # (Optional) DetectorParameters object
    )
    # the underscore _ is the rejected markers

    if ids is not None and len(ids) > 0:
        # Interpolate (insert) CharUco corners (refines marker positions)
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,      # Output from detectMarkers()
            markerIds=ids,              # Output from detectMarkers()
            image=grayed_frame,         # Grayscale image
            board=charuco_board,        # CharucoBoard object
            cameraMatrix=None,          # (Optional) Use for better accuracy
            distCoeffs=None             # (Optional) Lens distortion
        )
        # retval: number of corners detected
        # charucoCorners: defined 2D image positions of Charuco corners
        # charucoIds: their corresponding board IDs (1-50)

        # Draw the green boxes + optional IDs around detected ArUco tags (markers)
        aruco.drawDetectedMarkers(frame, corners, ids)
        if retval > 0:
            # Draw the red dots and optional IDs for refined Charuco corners (corners)
            aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    # Show the annotated image
    cv2.imshow("CharUco Capture", frame)

    # Press 's' to save the frame only if CharUco corners were found
    key = cv2.waitKey(1)
    if key == ord('s') and retval > 0:
        img_path = os.path.join(save_folder, f"trial {trial_number} | charuco_{img_id}.png")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        img_id += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
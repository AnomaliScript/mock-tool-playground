import cv2
import cv2.aruco as aruco
import os

# Trial number for .png organization
trial_number = input("Which trial is this? (int): ")

# Define CharUco board parameters
squares_x = 7  # number of squares in X direction
squares_y = 5  # number of squares in Y direction
square_length = 0.2  # in meters (used later for calibration)
marker_length = 0.15  # in meters (used later for calibration)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # Stable in OpenCV 4.6.0.66
charuco_board = aruco.CharucoBoard_create(
    squares_x,           # Number of squares along X (columns)
    squares_y,           # Number of squares along Y (rows)
    square_length,       # Size of a square (in meters or any consistent unit)
    marker_length,       # Size of each ArUco marker inside the squares
    aruco_dict           # The ArUco dictionary from Dictionary_get()
)

# Save folder
save_folder = "charuco_images/"

# Start camera
cap = cv2.VideoCapture(0)
img_id = 0

while True:
    ret, frame = cap.read()
    #frame = cv2.flip(frame, 1) The Forbidden Line
    grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(grayed_frame, aruco_dict, parameters = aruco.DetectorParameters_create())

    aruco.drawDetectedMarkers(grayed_frame, corners, ids)
    print(f"Markers seen: {len(corners)}")
    
    retval = 0
    if ids is not None and len(ids) > 0:
        # Interpolate (insert) CharUco corners (refines marker positions)
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,      # Output from detectMarkers()
            markerIds=ids,              # Output from detectMarkers()
            image=grayed_frame,         # Grayscale image
            board=charuco_board         # CharucoBoard object
        )
        # retval: number of corners detected
        # charucoCorners: defined 2D image positions of Charuco corners
        # charucoIds: their corresponding board IDs (1-50)

        # Draw the green boxes + optional IDs around detected ArUco tags (markers)
        aruco.drawDetectedMarkers(grayed_frame, corners, ids)
        if retval > 0:
            # Draw the red dots and optional IDs for refined Charuco corners (corners)
            aruco.drawDetectedCornersCharuco(grayed_frame, charuco_corners, charuco_ids)

    # Show the annotated image
    cv2.imshow("CharUco Capture", grayed_frame)

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
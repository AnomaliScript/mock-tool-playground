import cv2
import cv2.aruco as aruco

# Load ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters()

# Open camera
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
frame = cv2.flip(frame, 1)  # 1 = horizontal flip

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers
    aruco.drawDetectedMarkers(frame, corners, ids)

    # Print detected IDs
    if ids is not None:
        print("Detected IDs:", ids.flatten())

    # Show the image
    cv2.imshow('ArUco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
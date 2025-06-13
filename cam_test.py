import cv2
import cv2.aruco as aruco

print("OpenCV version:", cv2.__version__)

# Fix for older versions
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame.")
        break

    #frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    try:
        corners, ids, rejected = aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=parameters
        )
    except Exception as e:
        print("Detection error:", e)
        break

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        print("Detected IDs:", ids.flatten())

    cv2.imshow("Aruco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

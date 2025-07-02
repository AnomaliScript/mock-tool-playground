import cv2
import cv2.aruco as aruco
import numpy as np

print("OpenCV version:", cv2.__version__)

# Fix for older versions
parameters = aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

# Map tag IDs to tool names
tool_map = {
    0: "Scalpel",
    1: "Forceps",
    2: "Suction Tip",
    3: "Probe",
    4: "Camera Tool",
    # Add more as needed
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame.")
        break

    #frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    all_corners = []
    all_ids = []
    all_rejected = []

    # List of dictionaries you want to use
    dictionaries = [
        aruco.DICT_4X4_50,
        aruco.DICT_5X5_50,
        aruco.DICT_6X6_50,
        aruco.DICT_7X7_50
    ]

    for dict_id in dictionaries:
        aruco_dict = aruco.Dictionary_get(dict_id)
        try:
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                all_corners.extend(corners)
                all_ids.append(ids)
            if rejected is not None:
                all_rejected.extend(rejected)
        except Exception as e:
            print("Detection error:", e)
            break

    # Concatenate all ids into one numpy array
    if all_ids and len(all_ids) > 0:
        all_ids = np.concatenate(all_ids)
    else:
        all_ids = None
    
    if all_ids is not None:
        aruco.drawDetectedMarkers(frame, all_corners, all_ids)
        for tag_id in all_ids.flatten():
            tool_name = tool_map.get(tag_id, "Unknown Tool")
            print(f"Detected Tool: {tool_name} (Tag ID: {tag_id})")

    cv2.imshow("Aruco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

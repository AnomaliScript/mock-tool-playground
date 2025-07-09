import cv2
import cv2.aruco as aruco
from pupil_apriltags import Detector
import numpy as np

print("OpenCV version:", cv2.__version__)

# Initialize AprilTag detector (default uses tag36h11 family)
at_detector_25 = Detector(
    families='tag25h9',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.5,
    debug=0
)

at_detector_36 = Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.5,
    debug=0
)

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
    5: "Retractor",
    6: "Surgical Scissors",
    7: "Hemostat",
    8: "Needle Holder",
    9: "Electrocautery",
    10: "Bone Saw",
    11: "Clamp",
    12: "Tissue Forceps",
    13: "Speculum",
    14: "Dilator",
    15: "Towel Clamp",
    16: "Suture Passer",
    17: "Irrigation Syringe",
    18: "Stapler",
    19: "Curette",
    20: "Rongeur",
    21: "Periosteal Elevator",
    22: "Mallet",
    23: "Chisel",
    24: "Bone File",
    25: "Gauze Pad",
    26: "Sponge Stick",
    27: "Laparoscope",
    28: "Trocar",
    29: "Veress Needle",
    30: "Endoscopic Grasper",
    31: "Light Source"
    # et cetera
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
    april_ids = []
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
            # AprilTags first
            results_25 = at_detector_25.detect(gray)

            # Drawing
            for r in results_25:
                (ptA, ptB, ptC, ptD) = r.corners
                ptA = tuple(map(int, ptA))
                ptB = tuple(map(int, ptB))
                ptC = tuple(map(int, ptC))
                ptD = tuple(map(int, ptD))

                # Draw bounding box
                # Using np for drawing with polylines (more compact)
                pts = np.array([ptA, ptB, ptC, ptD], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Draw tag center
                (cX, cY) = tuple(map(int, r.center))
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

                # Draw tag ID
                tag_id = r.tag_id
                cv2.putText(
                    frame, 
                    str(tag_id), 
                    (ptA[0], ptA[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
                
                april_ids.append(tag_id)

            results_36 = at_detector_36.detect(gray)

            # Drawing
            for r in results_36:
                (ptA, ptB, ptC, ptD) = r.corners
                ptA = tuple(map(int, ptA))
                ptB = tuple(map(int, ptB))
                ptC = tuple(map(int, ptC))
                ptD = tuple(map(int, ptD))

                # Draw bounding box
                # Using np for drawing with polylines (more compact)
                pts = np.array([ptA, ptB, ptC, ptD], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Draw tag center
                (cX, cY) = tuple(map(int, r.center))
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

                # Draw tag ID
                tag_id = r.tag_id
                cv2.putText(
                    frame, 
                    str(tag_id), 
                    (ptA[0], ptA[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
                
                april_ids.append(tag_id)
            
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
        # Last arUco-related business
        aruco.drawDetectedMarkers(frame, all_corners, all_ids)
        
        # Combining arUco and AprilTag ids in a normal Python list
        all_ids = all_ids.tolist()
        for id in april_ids:
            all_ids.append(id)
        # Manual Flattening
        all_ids = [x[0] if isinstance(x, (list, np.ndarray)) else x for x in all_ids]
        for tag_id in all_ids:
            tool_name = tool_map.get(tag_id, "Tool not found")
            print(f"Detected Tool: {tool_name} (Tag ID: {tag_id})")

    cv2.imshow("Aruco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
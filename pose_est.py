import cv2
import numpy as np
from pupil_apriltags import Detector

# Load calibration
data = np.load("camera_calibration.npz")
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

# Define physical tag size (in meters)
tag_size = 0.015

# Real-world 3D coordinates of tag corners (same order as pupil_apriltags output)
obj_points = np.array([
    [-tag_size/2,  tag_size/2, 0],
    [ tag_size/2,  tag_size/2, 0],
    [ tag_size/2, -tag_size/2, 0],
    [-tag_size/2, -tag_size/2, 0]
], dtype=np.float32)

# Define tool lookup table
tool_map = {
    0: "Scalpel", 1: "Forceps", 2: "Suction Tip", 3: "Probe",
    4: "Camera Tool", 5: "Retractor", 6: "Surgical Scissors", 7: "Hemostat"
    # ... add more if needed
}

# Initialize AprilTag detector
at_detector = Detector(families='tag25h9')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)

    for tag in tags:
        corners = tag.corners.astype(np.float32)

        # Pose estimation
        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners, camera_matrix, dist_coeffs
        )

        rounded_pos = np.round(tvec.ravel(), 2)


        if success:
            cv2.drawFrameAxes(frame, 
                              camera_matrix, 
                              dist_coeffs, 
                              rvec, 
                              tvec, 
                              0.02)

            # Display tag info
            tool_name = tool_map.get(tag.tag_id, f"Unknown Tool {tag.tag_id}")
            cv2.putText(frame, 
                        f"{tool_name} (ID: {tag.tag_id}) pos: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}",
                        (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 100, 0), 
                        2)

    cv2.imshow("AprilTag Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
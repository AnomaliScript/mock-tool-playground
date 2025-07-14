import cv2
import numpy as np
from pupil_apriltags import Detector
from classes import ToolAdapter, Slots

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

# Initializing first ToolAdapter class
adapter = ToolAdapter(tool_map, 
                      6, 
                      None)

# Initialize AprilTag detector
at_detector = Detector(families='tag25h9')

# Function for checking for registered tags within view
def view_shown_tags(tags_param):
    tags_in_view = []
    adapter.available = []
    for tag in tags_param:
        if tag in adapter.available:
            tags_in_view.append(tag)

    for i in range(len(tags_in_view)):
        print(f"{i}: {adapter.available[tags_in_view[i]]}")
    
cap = cv2.VideoCapture(0)

# Dictionary saving tags with their theoretical respective positions
tag_positions = {}
# Dictionary saving tags with their actual respective positions
true_positions = {}

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
        tag_positions[tag], true_positions[tag] = rounded_pos

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

    # attach
    if cv2.waitKey(1) & 0xFF == ord('a'):
        # To be replaced
        view_shown_tags(tags)

        # Terminal UI :sob: (will implement dashboard-like app later)
        tool_id = input("Which tool would you like to attach?")

        while (tool_id.type() != int):
            print("Please type the number corresponding to the tool.")
            tool_id = input("Which tool would you like to attach?")

        new_pos = adapter.attach_tool(tool_id, tag_positions[tool_id], None)
        if isinstance(new_pos, (tuple, np.ndarray)):
            tag_positions[tool_id] = new_pos
        else:
            print(f"{new_pos}")
    # detach
    if cv2.waitKey(1) & 0xFF == ord('d'):
        # To be replaced
        view_shown_tags(tags)

        tool_id = input("Which tool would you like to detach?")

        while (tool_id.type() != int):
            print("Please type the number corresponding to the tool.")
            tool_id = input("Which tool would you like to detach?")

        adapter.detach_tool(tool_id, tag_positions[tool_id])
    # move
    if cv2.waitKey(1) & 0xFF == ord('m'):
        # To be replaced
        view_shown_tags(tags)

        tool_id = input("Which tool would you like to move?")
        
        while (tool_id.type() != int):
            print("Please type the number corresponding to the tool.")
            tool_id = input("Which tool would you like to move?")

        adapter.move_tool_to(tool_id, rounded_pos)
        # Saves the actual position (true_position) of the tag as its tag_position
        tag_positions[tool_id] = true_positions[tool_id]
        
    # get position
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # To be replaced
        view_shown_tags(tags)

        tool_id = input("Which tool would you like to get the position of?")

        while (isinstance(tool_id, int)):
            print("Please type the number corresponding to the tool.")
            tool_id = input("Which tool would you like to get the position of?")
        
        print(f"{rounded_pos}")
    # end condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
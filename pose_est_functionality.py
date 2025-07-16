import cv2
import numpy as np
from pupil_apriltags import Detector
from classes import ToolAdapter, Slots
from functions import get_tool_id

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

# Initializing first Slots class
slots = Slots(num_slots=6, center_pos=(0, 0, 0.5))

# Initializing first ToolAdapter class
adapter = ToolAdapter(available_tools=tool_map, 
                      holding_limit=6, 
                      possible_positions=None)

# Initialize AprilTag detector
at_detector = Detector(families='tag25h9')

# Function for checking for registered tags within view
def view_shown_tags(tags_param):
    tags_in_view = []
    for tag in tags_param:
        if tag.tag_id in adapter.available:
            tags_in_view.append(tag.tag_id)

    print(f"There are {len(tags_in_view)} tools in view.")

    for i in range(len(tags_in_view)):
        print(f"{i}: {adapter.available[tags_in_view[i]]}")
    
cap = cv2.VideoCapture(0)

# Dictionary saving tags with their actual respective positions
# This is useful to temporariy store the tag's positions until they are handed off to their calss attributes and whathot
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
        if not success:
            continue  # Skip if pose estimation failed

        rounded_pos = np.round(tvec.ravel(), 2)

        # Use tag.tag_id as the key
        true_positions[tag.tag_id] = rounded_pos
        print(f"Tool id: {tag.tag_id}")

        # Display virtual slots
        for pose in slots.slot_positions.values():
            x, y = int(round(pose[0])), int(round(pose[1]))
            cv2.circle(frame, center=(x, y), radius=10, color=(0, 255, 0), thickness=2)

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
        tool_id = get_tool_id("attach")

        print(f"{tool_map[tool_id]}")

        new_pos = adapter.attach_tool(chosen_id=tool_id, pose=true_positions[tool_id], target_pos=None, slots_obj=slots)
        if isinstance(new_pos, (tuple, np.ndarray)):
            true_positions[tool_id] = new_pos
        else:
            print(f"{new_pos}")
    # detach
    if cv2.waitKey(1) & 0xFF == ord('d'):
        # To be replaced
        view_shown_tags(tags)

        tool_id = get_tool_id("detach")

        adapter.detach_tool(tool_id, true_positions[tool_id])
    # move
    if cv2.waitKey(1) & 0xFF == ord('m'):
        # To be replaced
        view_shown_tags(tags)

        tool_id = get_tool_id("move")

        adapter.move_tool_to(tool_id, rounded_pos)
        # Saves the actual position (true_position) of the tag as its tag_position
        true_positions[tool_id] = true_positions[tool_id]
        
    # get position
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # To be replaced
        view_shown_tags(tags)

        tool_id = get_tool_id("get the position of")
        
        print(f"{rounded_pos}")
    # end condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
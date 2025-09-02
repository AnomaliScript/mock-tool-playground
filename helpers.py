def get_tmap(controller):
    return controller.shared_data["tool_map"]

def get_pmap(controller):
    return controller.shared_data["pos"]

def april_to_position(controller_param, april_id):
        # get the mapping
        pos_map = get_pmap(controller_param)

        # look up position ID if it exists, otherwise return None
        return pos_map.get(april_id, None)

def position_to_april(controller_param, pos_id):

        # look up april ID if it exists, otherwise return None
        return next((k for k, v in get_pmap(controller_param).items() if v == pos_id), None)

# Equation: width = height * aspect_ratio
def resize_to_fit_4_3(image, max_w, max_h):
    target_aspect = 4 / 3
    new_w = max_w
    new_h = int(max_w / target_aspect)
    if new_h > max_h:
        new_h = max_h
        new_w = int(max_h * target_aspect)
    if new_w > max_w:
        new_w = max_w
        new_h = int(max_w / target_aspect)
    return image.resize((new_w, new_h))

# VISION (wip)

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
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

# Open webcam globally
cap = cv2.VideoCapture(0)

detector = Detector(families="tag25h9")

# GLOBAL start/stop camera booleran
stop_camera = False

def update_video(self):
    # stop gate
    if stop_camera:
        return

    ret, frame = cap.read()
    if not ret:
        # camera hiccup: try again shortly
        self.after(100, self.update_video)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = self.detector.detect(gray)  # assumes self.detector exists

    current_seen = set()

    for tag in tags:
        tid = int(tag.tag_id)
        current_seen.add(tid)

        corners = tag.corners.astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners, camera_matrix, dist_coeffs
        )
        if not success:
            continue

        # axes
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.02)

        # tool name
        tm = get_tmap(self.controller)
        tool_name = tm.get(tid, f"Unknown Tool {tid}")

        # decide which ID to show
        # v2 helper for pos-id, but keep v1's "mode" & "center" behavior
        id_display = tid
        helping_text = "April_ID"

        if not self.controller.shared_data.get("show_april_mode", True):
            helping_text = "Position_ID"
            id_display = april_to_position(self.controller, tid)

        # center override: if the displayed ID equals the center ID
        center_id = self.controller.shared_data.get("center", "")
        if id_display == center_id:
            helping_text = "Center"
            tool_name = ""  # v1 behavior: hide name for center

        # label
        cv2.putText(
            frame,
            f"{tool_name} ({helping_text}: {id_display}) "
            f"pos: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}",
            (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 100, 0),
            2
        )

    # track visible IDs (from v1)
    self.visible_ids = current_seen

    # convert and fit (v1), with safe size discovery
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # find a good size to draw: prefer known label size; else ask widget; else fallback
    lw = getattr(self, "label_width", None)
    lh = getattr(self, "label_height", None)
    if not (lw and lh):
        try:
            lw = self.camera_label.winfo_width() or 640
            lh = self.camera_label.winfo_height() or 480
        except Exception:
            lw, lh = 640, 480

    resized = resize_to_fit_4_3(img, lw, lh)
    self.imgtk = ctk.CTkImage(light_image=resized, size=resized.size)
    self.camera_label.configure(image=self.imgtk)
    self.camera_label.image = self.imgtk  # keep a reference to prevent GC

    # schedule next tick (v2 style)
    self.after(15, self.update_video)
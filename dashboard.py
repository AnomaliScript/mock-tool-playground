import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
import threading
from pupil_apriltags import Detector
from collections import defaultdict, deque
import time
import classes
import helpers

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

# Set appearance mode and theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("themes/oceanix.json")

class BasePage(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.controller.shared_data.setdefault("show_april_mode", True)
        self.controller.shared_data["show_april_mode"] = True

class PreparationPage(BasePage):
    def __init__(self, master, controller):
        super().__init__(master, controller)

        # --- structure for the whole page
        cols, rows = 2, 3
        for c in range(cols):
            self.grid_columnconfigure(c, weight=1)
        for r in range(rows):
            self.grid_rowconfigure(r, weight=0)

        left = ctk.CTkFrame(self, fg_color="#333333")
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        right = ctk.CTkFrame(self, fg_color="#555555")
        right.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=10, pady=10)

        # tool defaults
        self.controller.shared_data.setdefault("storage", 6)
        self.controller.shared_data.setdefault("tool_map", {})
        self.controller.shared_data.setdefault("center", "")
        # position map (AprilTag: position)
        self.controller.shared_data.setdefault("pos", {})

        # --- left pane grid (2 cols x 5 rows: row1 "grows") ---
        cols, rows = 2, 5
        for c in range(cols):
            self.grid_columnconfigure(c, weight=1)
        for r in range(rows):
            self.grid_rowconfigure(r, weight=0)
        left.grid_columnconfigure(0, weight=1)
        left.grid_columnconfigure(1, weight=1)
        # left.grid_rowconfigure(0, weight=0)   # storage
        left.grid_rowconfigure(1, weight=1)   # Middle row expands (list/add)
        # left.grid_rowconfigure(2, weight=0)   # Center ID
        # left.grid_rowconfigure(3, weight=0)   # Position IDs
        # left.grid_rowconfigure(4, weight=0)   # Feedback
        # left.grid_rowconfigure(5, weight=0)   # Login

        # ========= row 0: Storage Limit =========
        storage = ctk.CTkFrame(left)
        storage.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        for c in range(3): storage.grid_columnconfigure(c, weight=1)

        ctk.CTkLabel(storage, text="Storage Limit", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6,4))

        self.storage_label = ctk.CTkLabel(
            storage, text=f"Current value: {self.controller.shared_data['storage']}"
        )
        self.storage_label.grid(row=1, column=0, sticky="w", padx=6)

        tool_slots_question = ctk.CTkLabel(storage, text="Enter how many tool slots are on the prototype")
        tool_slots_question.grid(row=1, column=1, sticky="w", padx=6)

        self.hold_num = ctk.CTkEntry(storage, placeholder_text="ex: 5")
        self.hold_num.grid(row=1, column=2, sticky="ew", padx=6)

        ctk.CTkButton(storage, text="Submit", command=self.submit_storage).grid(row=2, column=2, sticky="ew", padx=6, pady=(6,0))

        # ========= row 1-2 col 0: List of Tools =========
        tools_list = ctk.CTkFrame(left)
        tools_list.grid(row=1, rowspan=2, column=0, sticky="nsew", padx=10, pady=10)
        tools_list.grid_columnconfigure(0, weight=1)
        tools_list.grid_rowconfigure(1, weight=1)  # list grows

        ctk.CTkLabel(tools_list, text="Current tools available:", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))

        self.tool_list_frame = ctk.CTkScrollableFrame(tools_list)
        self.tool_list_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0,6))

        self.tool_list_text = ctk.CTkLabel(self.tool_list_frame, justify="left", anchor="w")
        self.tool_list_text.grid(row=0, column=0, sticky="w", padx=6, pady=6)

        # ========= row 1 col 1: Adding Tools =========
        add_tool = ctk.CTkFrame(left)
        add_tool.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        for c in range(1): add_tool.grid_columnconfigure(c, weight=1)

        add_tool = ctk.CTkLabel(add_tool, text="Add tool", font=("TkDefaultFont", 14, "bold"))
        add_tool.grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))

        ctk.CTkLabel(add_tool, text="Tool name").grid(row=1, column=0, sticky="w", padx=6)
        self.new_tool = ctk.CTkEntry(add_tool, placeholder_text="Retractor")
        self.new_tool.grid(row=2, column=0, sticky="ew", padx=6, pady=(0,8))

        ctk.CTkLabel(add_tool, text="AprilTag ID (optional)").grid(row=3, column=0, sticky="w", padx=6)
        self.new_tool_id = ctk.CTkEntry(add_tool, placeholder_text="12")
        self.new_tool_id.grid(row=4, column=0, sticky="ew", padx=6, pady=(0,8))


        ctk.CTkLabel(add_tool, text="Position ID (optional)").grid(row=5, column=0, sticky="w", padx=6)
        self.new_pos_id = ctk.CTkEntry(add_tool, placeholder_text="1")
        self.new_pos_id.grid(row=6, column=0, sticky="ew", padx=6, pady=(0,8))

        submit_new_tool = ctk.CTkButton(add_tool, text="Submit New Tool", command=self.submit_additional_tool)
        submit_new_tool.grid(row=7, column=0, sticky="ew", padx=6, pady=(0,6))

        self.new_tool.bind("<Return>", lambda _e: self.submit_additional_tool())
        self.new_tool_id.bind("<Return>", lambda _e: self.submit_additional_tool())

        # ========= row 2 col 1: Removing tools =========
        remove_tool = ctk.CTkFrame(left)
        remove_tool.grid(row=2, column=1, sticky="nsew", padx=10, pady=10)
        # for c in range(1): remove_tool.grid_columnconfigure(c, weight=1)

        remove_tool = ctk.CTkLabel(remove_tool, text="Remove tool", font=("TkDefaultFont", 14, "bold"))
        remove_tool.grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))

        ctk.CTkLabel(remove_tool, text="AprilTag ID").grid(row=1, column=0, sticky="w", padx=6)
        self.remove_tool = ctk.CTkEntry(remove_tool, placeholder_text="ex: 0")
        self.remove_tool.grid(row=2, column=0, sticky="ew", padx=6, pady=(0,8))

        submit_removal = ctk.CTkButton(remove_tool, text="Remove Tool", command=self.removing_tool)
        submit_removal.grid(row=5, column=0, sticky="ew", padx=6, pady=(0,6))

        self.remove_tool.bind("<Return>", lambda _e: self.remove_tool())
        self.remove_tool.bind("<Return>", lambda _e: self.remove_tool())

        # ========= row 3: Center ID =========
        center = ctk.CTkFrame(left)
        center.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        for c in range(3): center.grid_columnconfigure(c, weight=1) # 3 spots

        center_tag = ctk.CTkLabel(center, text="Center Tag", font=("TkDefaultFont", 14, "bold"))
        center_tag.grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6,4))

        self.center_label = ctk.CTkLabel(center, text=f"Current Center AprilTag ID: {self.controller.shared_data['center']}")
        self.center_label.grid(row=1, column=0, sticky="w", padx=6)

        self.center = ctk.CTkEntry(center, placeholder_text="ex: 0")
        self.center.grid(row=1, column=1, sticky="ew", padx=6)

        submit_center_id = ctk.CTkButton(center, text="Submit Center ID", command=self.submit_center)
        submit_center_id.grid(row=1, column=2, sticky="ew", padx=6)

        # ========= row 4: Feedback =========
        self.feedback = ctk.CTkLabel(left, text="", text_color="#FFCC66")
        self.feedback.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=6, pady=(2,4))

        # ========= row 5 col 0: Login =========
        login = ctk.CTkButton(left, text="Login", command=self.login)
        login.grid(row=5, column=0, sticky="ew", padx=10, pady=(0,10))

        # ========= row 5 col 1: Switch from April mode to Position mode (IDs) =========
        self.switch = ctk.CTkButton(left, 
                               text=f"Switch to {"April" if self.controller.shared_data["show_april_mode"] else "Position"} mode", 
                               command=self.switch)
        self.switch.grid(row=5, column=1, sticky="ew", padx=10, pady=(0,10))

        # --- right: camera ---
        self.stop_camera = False
        self.camera_label = ctk.CTkLabel(right, text="")
        self.camera_label.pack(pady=10, expand=True)

        self.detector = Detector(families="tag25h9")

        # initial render of the list
        self._render_tool_list()

    def _render_tool_list(self):
        tm = helpers.get_tmap(self.controller)
        pm = helpers.get_pmap(self.controller)

        lines = []
        for tid in sorted(tm.keys()):
            tool_name = tm[tid]
            pid = pm.get(tid, "<no pos>")   # safely get pid, fallback if not found
            lines.append(f"(AprilTag ID: {tid}, Pos ID: {pid}): {tool_name}")

        if not lines:
            lines = ["<no tools>"]

        self.tool_list_text.configure(text="\n".join(lines))

    def _display_feedback(self, msg: str, ok: bool = True):
        color = "#A4E8A2" if ok else "#FF8A80"
        self.feedback.configure(text=msg, text_color=color)

        # ---------- button callbacks (CTk) ----------
        name = (self.new_tool.get() or "").strip()
        id_text = (self.new_tool_id.get() or "").strip()

        if not name:
            # optionally show a small feedback label
            return

        tm = helpers.get_tmap(self.controller)

        # Parse optional ID
        tag_id = int(id_text) if id_text.isdigit() else None
        if tag_id is None:
            # auto-assign smallest unused
            used = set(tm.keys())
            tag_id = 0
            while tag_id in used:
                tag_id += 1
        else:
            if tag_id in tm:
                # collision: bail or choose next free
                # here we'll bail; alternatively, compute next free like above
                # show feedback if you have a label
                return

        tm[tag_id] = name

        # Refresh list
        lines = [f"{tid}: {tm[tid]}" for tid in sorted(tm.keys())]
        self.tool_list_text.configure(text="\n".join(lines) if lines else "<no tools>")

        # Clear inputs
        self.new_tool.delete(0, "end")
        self.new_tool_id.delete(0, "end")

    def submit_storage(self):
        val = (self.hold_num.get() or "").strip()
        if val.isdigit():
            self.controller.shared_data["storage"] = int(val)
            self.storage_label.configure(text=f"Current value: {val}")
            self._display_feedback("Storage limit updated.")
        else:
            self._display_feedback("Enter a whole number for storage limit.", ok=False)
        self.hold_num.delete(0, "end")

    def submit_additional_tool(self):
        name = (self.new_tool.get() or "").strip()
        id_text = (self.new_tool_id.get() or "").strip()
        pos_id_text = (self.new_pos_id.get() or "").strip()

        # basic validation
        if not name:
            self.feedback.configure(text="Tool name cannot be empty", text_color="#FF6666")
            return

        tool_map = helpers.get_tmap(self.controller)

        # parse/assign AprilTag ID
        if id_text:
            try:
                tool_id = int(id_text)
            except ValueError:
                self.feedback.configure(text="AprilTag ID must be a number", text_color="#FF6666")
                return
            if tool_id in tool_map:
                self.feedback.configure(text=f"AprilTag ID {tool_id} already exists", text_color="#FF6666")
                return
        else:
            tool_id = max(tool_map.keys(), default=-1) + 1

        # add to tool map
        tool_map[tool_id] = name

        pos_map = helpers.get_pmap(self.controller)
        if pos_id_text:
            try:
                pos_id = int(pos_id_text)
            except ValueError:
                self.feedback.configure(text="position ID must be a number", text_color="#FF6666")
                return

            # ensure uniqueness among existing position AprilTag IDs
            if pos_id in pos_map.values():
                self.feedback.configure(text=f"position ID {pos_id} already in use for another tool", text_color="#FF6666")
                return
        else:
            if tool_id not in pos_map.values():
                pos_id = tool_id
            else: 
                """
                auto-pick the smallest unused position id starting from 1 (increments up by one unitl it gets to a "vacant" spot)
                """
                used = set(pos_map.values())
                pos_id = 1
                while pos_id in used:
                    pos_id += 1

        # saving the tool's position id
        self.controller.shared_data["pos"][tool_id] = pos_id

        self._render_tool_list()

        self.new_tool.delete(0, "end")
        self.new_tool_id.delete(0, "end")
        self.new_pos_id.delete(0, "end")

        self.feedback.configure(
            text=f"Added '{name}' (ID {tool_id}, position AprilTag ID {pos_id})",
            text_color="#66FF66"
        )

    def removing_tool(self):
        id_text = self.remove_tool.get().strip()
        tool_map_alias = helpers.get_tmap(self.controller)

        # basic validation
        if not id_text:
            self.feedback.configure(text="Please specify the AprilTag ID", text_color="#FF6666")
            return

        try:
            tool_id = int(id_text)
        except ValueError:
            self.feedback.configure(text="AprilTag ID must be a number", text_color="#FF6666")
            return

        if tool_id not in helpers.get_tmap(self.controller):
            self.feedback.configure(text=f"AprilTag ID {tool_id} not found", text_color="#FF6666")
            return

        try:
            removed_name = tool_map_alias.pop(tool_id)
        except KeyError:
            self.feedback.configure(text="There is no tool with said ID", text_color="#FF6666")
            return

        self._render_tool_list()

        self.remove_tool.delete(0, "end")

        self.feedback.configure(text=f"Removed '{removed_name}' (ID {tool_id})", text_color="#66FF66")

    def submit_pos_ids(self, april_id):
        # {AprilTag ID: position ID}
        april = (self.new_pos_april.get() or "").strip()
        new_pos_id = (self.new_pos_id.get() or "").strip()

        if not april:
            self.feedback.configure(text="AprilTag ID cannot be empty", text_color="#FF6666")
            return
        if not new_pos_id:
            self.feedback.configure(text="position ID cannot be empty", text_color="#FF6666")
            return

        # try parsing IDs
        try:
            april_id = int(april)
        except ValueError:
            self.feedback.configure(text="AprilTag ID must be a number", text_color="#FF6666")
            return

        try:
            pos_id = int(new_pos_id)
        except ValueError:
            self.feedback.configure(text="position ID must be a number", text_color="#FF6666")
            return

        pos_map = helpers.get_pmap(self.controller)

        # collisions
        if april_id in pos_map:
            self.feedback.configure(text=f"AprilTag {april_id} already mapped to {pos_map[april_id]}", text_color="#FF6666")
            return
        if pos_id in pos_map.values():
            self.feedback.configure(text=f"position ID {pos_id} already in use", text_color="#FF6666")
            return

        # add mapping
        pos_map[april_id] = pos_id

        # refresh list
        lines = [f"{a} → {p}" for a, p in sorted(pos_map.items())]
        self.position_ids_text.configure(text="\n".join(lines) if lines else "<no positions>")

        # clear entries
        self.new_pos_april.delete(0, "end")
        self.new_pos_id.delete(0, "end")

        # success feedback
        self.feedback.configure(text=f"Added AprilTag ID {april_id} as ID {pos_id}", text_color="#66FF66")

    def submit_center(self):
        value = self.center.get().strip()

        # basic validation
        if not value:
            self.feedback.configure(text="Center ID cannot be empty", text_color="#FF6666")
            return

        try:
            center_id = int(value)
        except ValueError:
            self.feedback.configure(text="Center ID must be a number", text_color="#FF6666")
            return

        # check if the proposed center_id is already set to a tool
        tm = helpers.get_tmap(self.controller)
        if center_id in tm:
            self.feedback.configure(text=f"ID {center_id} is already in use", text_color="#FF6666")
            return

        # update shared data
        self.controller.shared_data["center"] = center_id

        # update label display
        self.center_label.configure(text=f"Current Center AprilTag ID: {center_id}")

        # clear entry
        self.center.delete(0, "end")

        self.feedback.configure(text=f"Center ID set to {center_id}", text_color="#66FF66")

    def switch(self):
        if self.controller.shared_data["show_april_mode"]:
            self.controller.shared_data["show_april_mode"] = False
        else: 
            self.controller.shared_data["show_april_mode"] = True
        self.switch.configure(text=f"Switch to {"Position" if self.controller.shared_data["show_april_mode"] else "April"} mode")

    # Called whenever this page is shown, which works well for displaying the camera
    def tkraise(self, aboveThis=None):
        self.stop_camera = False
        self.update_video()
        super().tkraise(aboveThis)

    def update_video(self):
        if self.stop_camera:
            return
    
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray)

            for tag in tags:
                corners = tag.corners.astype(np.float32)
                # Pose estimation
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corners, camera_matrix, dist_coeffs
                )
                if not success:
                    continue  # Skip if pose estimation failed

                if success:
                    cv2.drawFrameAxes(frame, 
                                    camera_matrix, 
                                    dist_coeffs, 
                                    rvec, 
                                    tvec, 
                                    0.02)

                    # Display tag info
                    tm = helpers.get_tmap(self.controller)
                    tool_name = tm.get(tag.tag_id, f"Unknown Tool {tag.tag_id}") # Unknown/Unmapped IDs will naturally be April IDs
                    id_display = tag.tag_id

                    # Checking if the ID is Center (important)
                    if (id_display == self.controller.shared_data["center"]):
                        helping_text = "Center"
                        tool_name = ""
                    else:
                        # Checking if the April Mode is on
                        if self.controller.shared_data["show_april_mode"]:
                            helping_text = "April_ID"
                        else:
                            helping_text = "Position_ID"
                            pm = helpers.get_pmap(self.controller)
                            id_display = pm.get(tag.tag_id, "N/A")

                    cv2.putText(frame, 
                                f"{tool_name} ({helping_text}: {id_display}) pos: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}",
                                (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 100, 0), 
                                2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))  # optional resize
            self.imgtk = ctk.CTkImage(light_image=img, size=(640, 480))
            self.camera_label.configure(image=self.imgtk)

        self.after(15, self.update_video)
    
    # Hiding the camera upon exiting this page
    def on_hide(self):
        # Stop webcam loop when page is hidden
        self.stop_camera = True

    def login(self):
        self.controller.show_frame("DashboardPage")







class DashboardPage(BasePage):
    def __init__(self, master, controller):
        super().__init__(master, controller)

        self.adapter = classes.ToolAdapter(
                    available_tools=helpers.get_tmap(self.controller),
                    storage_limit=self.controller.shared_data["storage"]
                    )
        
        self.detector = Detector(families="tag25h9")
        
        # Initialize tag history for velocity calculations
        self.tag_hist = defaultdict(lambda: deque(maxlen=10))
        
        # Store current velocities for visible tags
        self.current_velocities = {}
        self.selected_velocity_tag = None          # which tag’s velocity to show live
        self.velocity_updated_at = {}              # tid -> last v,w update time (seconds)
        self.velocity_stale_after = 0.30
        
        # UI: Cutting up the screen (dimensioning)
        for i in range(14):
            self.grid_columnconfigure(i, weight=1)
        for i in range(6):
            self.grid_rowconfigure(i, weight=1)

        # ALSO UI: Background Creation (originally for debugging, but I want to keep it tbh)
        cols = 14
        rows = 6

        colors = [
            "#FFBC9A", "#FF9D6C", "#F9785C", "#C94F64",
            "#5A4C7A", "#3A3F66", "#222C4C", "#101B32",
            "#1C2D50", "#3A4D6C", "#637DA1", "#8DAFD1",
            "#D4D8E2", "#FFF2BF"
        ]

        # Configure the full grid
        for col in range(cols):
            self.grid_columnconfigure(col, weight=1)
        for row in range(rows):
            self.grid_rowconfigure(row, weight=1)

        # Create colorful background
        for row in range(rows):
            for col in range(cols):
                label = ctk.CTkLabel(
                    self,
                    text=f"{row},{col}",
                    fg_color=colors[col % len(colors)],
                    corner_radius=0,
                    text_color="black"
                )
                label.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)

        # Text Widgets (6x6 tiles covered up)
        # adapter_obj Dynamic CTk Frame; API functions that use this CTkFrame are defined further below
        self.panel = ctk.CTkFrame(self, fg_color="#FFBC9A")  # same color as before
        self.panel.grid(row=0, column=8, rowspan=2, columnspan=2, sticky="nsew", padx=10, pady=10)

        # optional: a title in the panel
        self.panel_title = ctk.CTkLabel(self.panel, text="Dashboard Panel", font=("TkDefaultFont", 16, "bold"))
        self.panel_title.pack(pady=(8, 6))

        self.emergency_stop = ctk.CTkFrame(self, fg_color="#FF0000")  # same color as before
        self.emergency_stop.grid(row=2, column=8, rowspan=2, columnspan=3, sticky="nsew", padx=10, pady=10)

        # Output for API methods
        self.panel_body = ctk.CTkFrame(self.panel, fg_color="transparent")
        self.panel_body.pack(fill="both", expand=True, padx=10, pady=10)

        # Available Tools List
        self.available_list = ctk.CTkLabel(self, fg_color="#F9785C", text="", anchor="center")
        self.available_list.grid(row=2, column=11, rowspan=2, columnspan=3, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.available_list, text="Current tools available:", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))

        self.available_list_frame = ctk.CTkScrollableFrame(self.available_list, fg_color="#F9785C")
        self.available_list_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        self.available_list_frame_text = ctk.CTkLabel(self.available_list_frame, fg_color="#F9785C", justify="left", anchor="w")
        self.available_list_frame_text.grid(row=0, column=0, sticky="w", padx=6, pady=6)

        # Current Tool Description (velocity included)
        self.current_tool = ctk.CTkLabel(self, fg_color="#5A4C7A", text="", anchor="center")
        self.current_tool.grid(row=0, column=10, rowspan=2, columnspan=4, sticky="nsew", padx=10, pady=10)

        # API Function Widget
        self.functions = ctk.CTkFrame(self, fg_color="#B4B4B4")
        self.functions.grid(row=4, column=8, rowspan=2, columnspan=6, sticky="nsew", padx=10, pady=10)

        # API Function Widget Construction
        api_cols = 5
        api_rows = 1
        for col in range(api_cols):
            self.functions.grid_columnconfigure(col, weight=1)
        for row in range(api_rows):
            self.functions.grid_rowconfigure(row, weight=1)

        # Widgets/Buttons within API Function Widget
        show_button = ctk.CTkButton(self.functions, text="Show All Available Tools", fg_color="#DDDDDD", 
                                    command=lambda: self._panel_show("Seen Tags", self.view_shown_tags))
        show_button.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.visible_ids = set()

        attach_button = ctk.CTkButton(self.functions, text="Attach Tool", fg_color="#CCCCCC", 
                                      command=lambda: self._panel_show("Seen Tags", self.attach))
        attach_button.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        detach_button = ctk.CTkButton(self.functions, text="Detach Tool", fg_color="#BBBBBB", 
                                      command=lambda: self._panel_show("Seen Tags", self.detach_tool))
        detach_button.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        velocity_check = ctk.CTkButton(self.functions, text="Check Velocity", fg_color="#AAAAAA", 
                                      command=lambda: self._panel_show("Seen Tags", self.check_velocity))
        velocity_check.grid(row=0, column=3, sticky="nsew", padx=5, pady=5)

        settings_button = ctk.CTkButton(self.functions, text="Settings Page", fg_color="#999999", 
                                        command=lambda: self.controller.show_frame("PreparationPage"))
        settings_button.grid(row=0, column=4, sticky="nsew", padx=5, pady=5)

        # Column Protect (minimum width)
        for i in [8, 9, 10, 11, 12, 13]:
            self.grid_columnconfigure(i, minsize=155)  # ensures at least 300px

        # Camera Time! (6x8 tiles covered up)
        # DISPLAY RESOLUTION: 4x3
        self.camera_label = ctk.CTkLabel(self, anchor="center", text="")
        self.camera_label.grid(row=0, column=0, rowspan=6, columnspan=8, sticky="nsew", padx=40, pady=40)

        self.stop_camera = True  # Start stopped
        self.current_image = None

        # Bind resize event
        self.camera_label.bind("<Configure>", self.on_resize)

        # Start update loop in thread
        self.update_thread = threading.Thread(target=self.update_video, daemon=True)
        self.update_thread.start()

    def _render_dshb_tool_list(self):
        tm = helpers.get_tmap(self.controller)
        pm = helpers.get_pmap(self.controller)

        lines = []
        for tid in sorted(tm.keys()):
            tool_name = tm[tid]
            pid = pm.get(tid, "<no pos>")
            lines.append(f"(AprilTag ID: {tid}, Pos ID: {pid}): {tool_name}")

        if not lines:
            lines = ["<no tools>"]

        self.available_list_frame_text.configure(text="\n".join(lines))

    def _panel_clear(self):
        for w in self.panel_body.winfo_children():
            w.destroy()

    def _panel_show(self, title: str, builder):
        self.panel_title.configure(text=title)
        self._panel_clear()
        builder(self.panel_body)  # builder represents the API function

    # Helpers (CTk-safe)
    
    # def _get_position_id(self, april_id: int):
    #     """
    #     Given an AprilTag ID, return the position position ID (if mapped).
    #     Falls back to the raw AprilTag ID if no mapping exists.
    #     """
    #     pos_map = helpers.get_pmap(self.controller)
    #     return pos_map.get(april_id, april_id)

    # API FUNCTIONS (add, remove, etc)

    def view_shown_tags(self, parent):
        tm = helpers.get_tmap(self.controller)
        ids = sorted(self.visible_ids)

        title = ctk.CTkLabel(parent, text="Seen Tags (this session):", anchor="w", justify="center")
        title.pack(anchor="w", pady=(0, 6))

        if not ids:
            ctk.CTkLabel(parent, text="<no tags detected yet>", anchor="w").pack(anchor="w")
            return

        lines = [f"{tid}: {tm.get(tid, f'Unknown Tool {tid}')}" for tid in ids]
        ctk.CTkLabel(parent, text="\n".join(lines), anchor="w", justify="left").pack(anchor="w")

    # Attaching, Part 1
    def attach(self, parent):
        
        title = ctk.CTkLabel(parent, text="Attach Tool", anchor="w", justify="center")
        title.pack(anchor="w", pady=(0, 6))

        # April Case
        if (self.controller.shared_data["show_april_mode"]):
            self.id2b_attached = ctk.CTkEntry(parent, placeholder_text="April ID")
            self.id2b_attached.pack(anchor="w", pady=(0, 12))
        # Preferred Case
        else:
            self.id2b_attached = ctk.CTkEntry(parent, placeholder_text="Preferred ID")
            self.id2b_attached.pack(anchor="w", pady=(0, 12))

        # Creating a label that attach_operation can refer to
        self.attach_feedback = ctk.CTkLabel(parent, text="", anchor="w", justify="center")
        self.attach_feedback.pack(anchor="w", pady=(0, 6))

        ctk.CTkButton(parent, text="Attach", command=lambda: self.attach_operation()).pack(anchor="w", pady=(0, 12))
        
    # Attaching, Part 2
    def attach_operation(self):
        tm = helpers.get_tmap(self.controller)
        id = self.id2b_attached.get().strip()

        # Checking if ID alr exsits
        if "name" in self.adapter.attached and self.adapter.attached["name"]:
            self.attach_feedback.configure(text="A tool is already attached", text_color="#FF6666")
            return
        # Checking if ID is an integer
        if id:
            try:
                converted_id = int(id)
            except ValueError:
                self.attach_feedback.configure(text="ID must be a number", text_color="#FF6666")
                return
        else:
            self.attach_feedback.configure(text="Please type in a number", text_color="#FF6666")
            return
        
        # TROUBLESHOOTING
        # print(f"{tm}")
        # print(f"{id}")
        # print(f"{helpers.position_to_april(self.controller, converted_id)}")
        # print(f"{self.controller.shared_data["show_april_mode"]}")
        # print(f"second case: {tm.get(helpers.position_to_april(self.controller, converted_id), None) == None}")
        # print(f"final: {(self.controller.shared_data["show_april_mode"] == False) and (tm.get(helpers.position_to_april(self.controller, converted_id), None) == None)}")
        # print(f"dne (april): {tm.get(converted_id, None) == None}, dne (pos): {(tm.get(helpers.position_to_april(self.controller, converted_id), None) == None)}")
        
        # Checking if the ID is mapped to a tool (April and Positoin cases, respectively)
        if ((self.controller.shared_data["show_april_mode"] and (tm.get(converted_id, None) == None)) or 
            (self.controller.shared_data["show_april_mode"] == False) and (helpers.position_to_april(self.controller, converted_id) == None)):
            self.attach_feedback.configure(text="Invalid ID", text_color="#FF6666")
            return
        
        # April Case
        if (self.controller.shared_data["show_april_mode"]):
            self.adapter.attached = {
                "pos_id" : helpers.april_to_position(self.controller, converted_id),
                "april_id" : converted_id,
                "name": tm.get(converted_id)
            }
            self.attach_feedback.configure(text=f"ID {self.adapter.attached["name"]} attached", text_color="#66FF66")
        # Position Case
        else: 
            april_id_version = helpers.position_to_april(self.controller, converted_id)
            self.adapter.attached = {
                "pos_id" : converted_id,
                "april_id" : april_id_version,
                "name": tm.get(april_id_version)
            }
            self.attach_feedback.configure(text=f"ID {self.adapter.attached["name"]} attached", text_color="#66FF66")

        print(f"{self.adapter.attached}")

    def detach_tool(self, parent):
        # Delete
        print(f"{self.adapter.attached}")
        ctk.CTkButton(parent, text="Attach", command=self.detach_operation()).pack(anchor="w", pady=(0, 12))

    def detach_operation(self):
        # TODO: employ motors and sensors to detach the tool
        self.adapter.attached = {}

    # Velocity Evaluation
    def check_velocity(self, parent):
        title = ctk.CTkLabel(parent, text="Check Velocity", anchor="w", justify="center")
        title.pack(anchor="w", pady=(0, 6))

        # Create input field for tag ID
        if (self.controller.shared_data["show_april_mode"]):
            self.velocity_tag_id = ctk.CTkEntry(parent, placeholder_text="April ID")
            self.velocity_tag_id.pack(anchor="w", pady=(0, 12))
        else:
            self.velocity_tag_id = ctk.CTkEntry(parent, placeholder_text="Position ID")
            self.velocity_tag_id.pack(anchor="w", pady=(0, 12))

        # Create button to check velocity
        ctk.CTkButton(parent, text="Check Velocity", command=lambda: self.retrieve_velocity()).pack(anchor="w", pady=(0, 12))
        ctk.CTkButton(parent, text="Stop Tracking", command=lambda: self._stop_velocity_follow()).pack(anchor="w", pady=(0, 12))

    # Velocity Calculation (ahhh trig)
    def rvec_to_R(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        return R

    def calc_angular_velocity(self, R_prev, R_cur, dt):
        """
        Approx angular velocity (rad/s) using matrix log of relative rotation.
        omega_vec points along rotation axis, magnitude = angular speed.
        """
        R_delta = R_cur @ R_prev.T
        # clamp numerical errors
        tr = np.clip((np.trace(R_delta) - 1) / 2.0, -1.0, 1.0)
        angle = np.arccos(tr)
        if dt <= 1e-6 or angle < 1e-6:
            return np.zeros(3)
        # rotation axis from skew-symmetric part
        w = (R_delta - R_delta.T) / (2*np.sin(angle))
        axis = np.array([w[2,1], w[0,2], w[1,0]])  # (wx, wy, wz)
        return axis * (angle / dt)  # rad/s

    def organize_velocity_data(self, tag_id: int, rvec, tvec):
        """
        Returns (linear_vel_mps[3], angular_vel_radps[3]) in the camera frame.
        Units assume your obj_points are in meters -> tvec is meters.
        """
        now = time.time()
        R = self.rvec_to_R(rvec)
        hist = self.tag_hist[tag_id]

        # If we have a previous sample, compute finite differences
        if len(hist) > 0:
            t_prev, p_prev, R_prev = hist[-1]
            dt = now - t_prev
            if dt > 1e-3:
                v = (tvec.reshape(3) - p_prev) / dt                    # m/s in camera frame
                w = self.calc_angular_velocity(R_prev, R, dt)          # rad/s
                # Debug: print velocity when it's non-zero
                if np.linalg.norm(v) > 0.001:  # Only print if velocity > 1mm/s
                    print(f"Tag {tag_id}: v={v}, w={w}, dt={dt:.3f}s")
            else:
                v = np.zeros(3); w = np.zeros(3)
        else:
            v = np.zeros(3); w = np.zeros(3)

        # push current sample
        hist.append((now, tvec.reshape(3), R))
        
        # Store current velocity for this tag
        self.current_velocities[tag_id] = (v, w)
        self.velocity_updated_at[tag_id] = now

    def retrieve_velocity(self):
        """Display velocity for a specific tag using current velocity data"""
        tag_id_input = self.velocity_tag_id.get().strip()
        
        # Checking for silly goober IDs
        if not tag_id_input:
            self.current_tool.configure(text="Please enter a tag ID")
            return
            
        try:
            tag_id = int(tag_id_input)
        except ValueError:
            self.current_tool.configure(text="Tag ID must be a number")
            return
        
        # Check if tag exists in tool map
        tm = helpers.get_tmap(self.controller)
        if tag_id not in tm:
            self.current_tool.configure(text=f"Tag ID {tag_id} not found in tool map")
            return
        
        # Check if tag is currently visible on screen
        if tag_id not in self.visible_ids:
            tool_name = tm[tag_id]
            self.current_tool.configure(text=f"{tool_name} (ID: {tag_id})\nTag not currently visible on screen")
            return
        
        # Check if we have current velocity data for this tag
        if tag_id not in self.current_velocities:
            tool_name = tm[tag_id]
            self.current_tool.configure(text=f"{tool_name} (ID: {tag_id})\nNo velocity data available yet")
            return
        
        # Get current velocity data and display it
        v, w = self.current_velocities[tag_id]
        self._display_velocity_data(tag_id, v, w)

    def _display_velocity_data(self, tag_id: int, v: np.ndarray, w: np.ndarray):
        """Helper method to display formatted velocity data"""
        # Check if tag exists in tool map
        tm = helpers.get_tmap(self.controller)
        if tag_id not in tm:
            self.current_tool.configure(text=f"Tag ID {tag_id} not found in tool map")
            return
        
        # Get tool name
        tool_name = tm[tag_id]
        
        # Format velocity data in an organized manner
        vx, vy, vz = v
        wx, wy, wz = w
        
        velocity_text = f"{tool_name} (ID: {tag_id})\n"
        velocity_text += f"Linear Velocity: ({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s\n"
        velocity_text += f"Angular Velocity: ({wx:.3f}, {wy:.3f}, {wz:.3f}) rad/s\n"
        velocity_text += f"Speed: {np.linalg.norm(v):.3f} m/s"
        
        self.current_tool.configure(text=velocity_text)

    def _stop_velocity_follow(self):
        self.selected_velocity_tag = None
        self.current_tool.configure(text="Velocity tracking cleared.")

    # MAIN DASHBOARD FUNCTIONS
    
    def on_resize(self, event):
        # Handle window resize and update image to match new label size
        self.label_width = event.width
        self.label_height = event.height

    def tkraise(self, aboveThis=None):
        self.stop_camera = False
        self.update_video()
        super().tkraise(aboveThis)

    def update_video(self): # Primary use is to update video, used for other things also
        self._render_dshb_tool_list() # Used here to refresh the available tools

        if self.stop_camera:
            return

        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray)

            # Keep track of seen tags
            current_seen = set()

            for tag in tags:
                tid = int(tag.tag_id)
                current_seen.add(tid)

                corners = tag.corners.astype(np.float32)
                # Pose estimation
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corners, camera_matrix, dist_coeffs
                )
                if not success:
                    continue  # Skip if pose estimation failed

                if success:

                    cv2.drawFrameAxes(frame, 
                                    camera_matrix, 
                                    dist_coeffs, 
                                    rvec, 
                                    tvec, 
                                    0.02)

                    # VELOCITY
                    self.organize_velocity_data(tid, rvec, tvec)

                    # Display tag info
                    tm = helpers.get_tmap(self.controller)
                    tool_name = tm.get(tag.tag_id, f"Unknown Tool {tag.tag_id}") # Unknown/Unmapped IDs will naturally be April IDs
                    id_display = tag.tag_id

                    # Checking if the ID is Center (important)
                    if (id_display == self.controller.shared_data["center"]):
                        helping_text = "Center"
                        tool_name = ""
                    else:
                        # Checking if the April Mode is on
                        if self.controller.shared_data["show_april_mode"]:
                            helping_text = "April_ID"
                        else:
                            helping_text = "Position_ID"
                            pm = helpers.get_pmap(self.controller)
                            id_display = pm.get(tag.tag_id, "N/A")

                    cv2.putText(frame, 
                                f"{tool_name} ({helping_text}: {id_display}) pos: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}",
                                (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 100, 0), 
                                2)

            self.visible_ids = current_seen

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Wait until label is laid out
            if hasattr(self, "label_width") and hasattr(self, "label_height"):
                resized_img = helpers.resize_to_fit_4_3(img, self.label_width, self.label_height)
                self.imgtk = ctk.CTkImage(light_image=resized_img, size=resized_img.size)
                self.camera_label.configure(image=self.imgtk)

            # Update label image
            self.camera_label.configure(image=self.imgtk)

        tid = self.selected_velocity_tag
        if tid is not None and tid in self.current_velocities:
            v, w = self.current_velocities[tid]
            self._display_velocity_data(tid, v, w)

        # Schedule next frame update ~60 FPS
        self.camera_label.after(15, self.update_video)

    def on_hide(self):
        # Stop webcam loop when page is hidden
        self.stop_camera = True


# App Controller
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("2100x900")

        # Create the container frame first
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)

        # Now configure grid responsiveness on the container
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.shared_data = {}
        # storage
        # tool_map
        # center
        # show_april_mode (boolean)
        # revisit_settings (boolean)
        # pos

        # Load page frames into self.container
        self.frames = {}
        for PageClass in (PreparationPage, DashboardPage):
            page_name = PageClass.__name__
            frame = PageClass(self.container, self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("PreparationPage")

    def show_frame(self, name):
        for frame in self.frames.values():
            if hasattr(frame, "on_hide"):
                frame.on_hide()
        frame = self.frames[name]
        frame.tkraise()

    def on_closing(self):
        cap.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
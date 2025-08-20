import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from pupil_apriltags import Detector
import classes

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

# Open webcam globally
cap = cv2.VideoCapture(0)

# Set appearance mode and theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("themes/oceanix.json")

class BasePage(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller

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
        self.controller.shared_data.setdefault("tool_map", {
            0: "Scalpel", 1: "Forceps", 2: "Suction Tip", 3: "Probe",
            4: "Camera Tool", 5: "Retractor", 6: "Surgical Scissors", 7: "Hemostat"
        })
        self.controller.shared_data.setdefault("center", "")

        # --- left pane grid (2 cols x 5 rows: row1 "grows") ---
        cols, rows = 2, 4
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
        # left.grid_rowconfigure(4, weight=0)   # Login

        # ========= row 0: storage Limit (colspan=2) =========
        storage = ctk.CTkFrame(left)
        storage.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        for c in range(3): storage.grid_columnconfigure(c, weight=1)

        ctk.CTkLabel(storage, text="storage Limit", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6,4))

        self.storage_label = ctk.CTkLabel(
            storage, text=f"Current value: {self.controller.shared_data['storage']}"
        )
        self.storage_label.grid(row=1, column=0, sticky="w", padx=6)

        tool_slots_question = ctk.CTkLabel(storage, text="Enter how many tool slots are on the prototype")
        tool_slots_question.grid(row=1, column=1, sticky="w", padx=6)

        self.hold_num = ctk.CTkEntry(storage, placeholder_text="e.g. 5")
        self.hold_num.grid(row=1, column=2, sticky="ew", padx=6)

        ctk.CTkButton(storage, text="Submit", command=self.submit_storage)\
            .grid(row=2, column=2, sticky="ew", padx=6, pady=(6,0))

        # ========= row 1 col 0: List of Tools =========
        tools_list = ctk.CTkFrame(left)
        tools_list.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
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

        ctk.CTkLabel(add_tool, text="Tool ID (optional)").grid(row=3, column=0, sticky="w", padx=6)
        self.new_tool_id = ctk.CTkEntry(add_tool, placeholder_text="12")
        self.new_tool_id.grid(row=4, column=0, sticky="ew", padx=6, pady=(0,8))

        submit_new_tool = ctk.CTkButton(add_tool, text="Submit New Tool", command=self.submit_additional_tool)
        submit_new_tool.grid(row=5, column=0, sticky="ew", padx=6, pady=(0,6))

        self.feedback = ctk.CTkLabel(add_tool, text="", text_color="#FFCC66")
        self.feedback.grid(row=6, column=0, sticky="w", padx=6, pady=(2,0))

        self.new_tool.bind("<Return>", lambda _e: self.submit_additional_tool())
        self.new_tool_id.bind("<Return>", lambda _e: self.submit_additional_tool())

        # ========= row 2: Center ID =========
        center = ctk.CTkFrame(left)
        center.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        for c in range(3): center.grid_columnconfigure(c, weight=1) # 3 spots

        center_tag = ctk.CTkLabel(center, text="Center Tag", font=("TkDefaultFont", 14, "bold"))
        center_tag.grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6,4))

        self.center_label = ctk.CTkLabel(center, text=f"Current center ID: {self.controller.shared_data['center']}")
        self.center_label.grid(row=1, column=0, sticky="w", padx=6)

        self.center = ctk.CTkEntry(center, placeholder_text="e.g. 0")
        self.center.grid(row=1, column=1, sticky="ew", padx=6)

        submit_center_id = ctk.CTkButton(center, text="Submit Center ID", command=self.submit_center)
        submit_center_id.grid(row=1, column=2, sticky="ew", padx=6)

        # ========= row 3 col 0: Position IDs =========
        # position_ids = ctk.CTkFrame(left)
        # position_ids.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        # position_ids.grid_columnconfigure(0, weight=1)
        # position_ids.grid_rowconfigure(1, weight=1)  # list grows

        # ctk.CTkLabel(position_ids, text="Current Positions Registered:", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))

        # self.position_ids_frame = ctk.CTkScrollableFrame(position_ids)
        # self.position_ids_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0,6))

        # self.position_ids_text = ctk.CTkLabel(self.position_ids_frame, justify="left", anchor="w")
        # self.position_ids_text.grid(row=0, column=0, sticky="w", padx=6, pady=6)

        # # ========= row 3 col 1: Position ID Assignment =========
        # position_assignment = ctk.CTkFrame(left)
        # position_assignment.grid(row=3, column=1, sticky="nsew", padx=10, pady=10)
        # position_assignment.grid_columnconfigure(0, weight=1)

        # ctk.CTkLabel(position_assignment, text="Add a Position ID", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))

        # ctk.CTkLabel(position_assignment, text="AprilTag ID").grid(row=1, column=0, sticky="w", padx=6)
        # self.new_pos_april = ctk.CTkEntry(position_assignment, placeholder_text="37")
        # self.new_pos_april.grid(row=2, column=0, sticky="ew", padx=6, pady=(0,8))

        # ctk.CTkLabel(position_assignment, text="Preferred Position ID").grid(row=3, column=0, sticky="w", padx=6)
        # self.new_pref_id = ctk.CTkEntry(position_assignment, placeholder_text="1")
        # self.new_pref_id.grid(row=4, column=0, sticky="ew", padx=6, pady=(0,8))

        # submit_new_pos = ctk.CTkButton(position_assignment, text="Submit New Position", command=self.submit_pref_ids)
        # submit_new_pos.grid(row=5, column=0, sticky="ew", padx=6, pady=(0,6))

        # self.feedback = ctk.CTkLabel(position_assignment, text="", text_color="#FFCC66")
        # self.feedback.grid(row=6, column=0, sticky="w", padx=6, pady=(2,0))

        # self.new_pos_april.bind("<Return>", lambda _e: self.submit_pref_ids())
        # self.new_pref_id.bind("<Return>", lambda _e: self.submit_pref_ids())
        
        # ========= (new) row 3: Login =========
        login = ctk.CTkButton(left, text="Login", command=self.login)
        login.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0,10))

        # --- right: camera ---
        self.stop_camera = False
        self.camera_label = ctk.CTkLabel(right, text="")
        self.camera_label.pack(pady=10, expand=True)

        self.detector = Detector(families="tag25h9")

        # initial render of the list
        self._render_tool_list()

    # Helpers (CTk-safe)
    def _tool_map(self):
        return self.controller.shared_data["tool_map"]

    def _render_tool_list(self):
        tm = self._tool_map()
        lines = [f"{tid}: {tm[tid]}" for tid in sorted(tm.keys())]
        if not lines:
            lines = ["<no tools>"]
        self.tool_list_text.configure(text="\n".join(lines))

    # def _smallest_unused_id(self):
    #     used = set(self._tool_map().keys())
    #     i = 0
    #     while i in used:
    #         i += 1
    #     return i

    def _parse_tool_entry(self, s: str):
        """
        Accepts:
          - 'Scalpel' -> ('Scalpel', None)
          - 'Scalpel:12' / 'Scalpel=12' / 'Scalpel 12' -> ('Scalpel', 12) if 12 is int
        """
        raw = (s or "").strip()
        if not raw:
            return None, None
        for sep in (":", "=", " "):
            if sep in raw:
                name, id_str = raw.split(sep, 1)
                name, id_str = name.strip(), id_str.strip()
                if not name:
                    return None, None
                return (name, int(id_str)) if id_str.isdigit() else (name, None)
        return raw, None

    def _display_feedback(self, msg: str, ok: bool = True):
        color = "#A4E8A2" if ok else "#FF8A80"
        self.feedback.configure(text=msg, text_color=color)

        # ---------- button callbacks (CTk) ----------
        name = (self.new_tool.get() or "").strip()
        id_text = (self.new_tool_id.get() or "").strip()

        if not name:
            # optionally show a small feedback label
            return

        tm = self.controller.shared_data["tool_map"]

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

    def _april_to_preferred(self, april_id):
        # get the mapping
        pref_map = self.controller.shared_data.get("pos", {})

        # look up preferred ID if it exists, otherwise fall back to raw
        return pref_map.get(april_id, april_id)

    def submit_storage(self):
        val = (self.hold_num.get() or "").strip()
        if val.isdigit():
            self.controller.shared_data["storage"] = int(val)
            self.storage_label.configure(text=f"Current value: {val}")
            self._display_feedback("Storage limit updated.")
        else:
            self._display_feedback("Enter a whole number for storage limit.", ok=False)

    def submit_additional_tool(self):
        name = self.new_tool.get().strip()
        id_text = self.new_tool_id.get().strip()

        # basic validation
        if not name:
            self.feedback.configure(text="Tool name cannot be empty", text_color="#FF6666")
            return

        # try parsing ID (optional)
        tool_map = self.controller.shared_data["tool_map"]
        if id_text:
            try:
                tool_id = int(id_text)
            except ValueError:
                self.feedback.configure(text="Tool ID must be a number", text_color="#FF6666")
                return
            if tool_id in tool_map:
                self.feedback.configure(text=f"Tool ID {tool_id} already exists", text_color="#FF6666")
                return
        else:
            # auto-pick the next free ID
            tool_id = max(tool_map.keys(), default=-1) + 1

        # add tool to map
        tool_map[tool_id] = name

        # refresh tool list display
        self._render_tool_list()

        # clear entries
        self.new_tool.delete(0, "end")
        self.new_tool_id.delete(0, "end")

        # success feedback
        self.feedback.configure(text=f"Added '{name}' (ID {tool_id})", text_color="#66FF66")

    def submit_pref_ids(self, april_id):
        # {AprilTag ID: Preferred ID}
        april = (self.new_pos_april.get() or "").strip()
        new_pref_id = (self.new_pref_id.get() or "").strip()

        # basic validation
        if not april:
            self.feedback.configure(text="AprilTag ID cannot be empty", text_color="#FF6666")
            return
        if not new_pref_id:
            self.feedback.configure(text="Preferred ID cannot be empty", text_color="#FF6666")
            return

        # try parsing IDs
        try:
            april_id = int(april)
        except ValueError:
            self.feedback.configure(text="AprilTag ID must be a number", text_color="#FF6666")
            return

        try:
            pref_id = int(new_pref_id)
        except ValueError:
            self.feedback.configure(text="Preferred ID must be a number", text_color="#FF6666")
            return

        # position map (AprilTag: Preferred)
        pref_map = self.controller.shared_data.setdefault("pos", {})

        # collisions
        if april_id in pref_map:
            self.feedback.configure(text=f"AprilTag {april_id} already mapped to {pref_map[april_id]}", text_color="#FF6666")
            return
        if pref_id in pref_map.values():
            self.feedback.configure(text=f"Preferred ID {pref_id} already in use", text_color="#FF6666")
            return

        # add mapping
        pref_map[april_id] = pref_id

        # refresh list
        lines = [f"{a} → {p}" for a, p in sorted(pref_map.items())]
        self.position_ids_text.configure(text="\n".join(lines) if lines else "<no positions>")

        # clear entries
        self.new_pos_april.delete(0, "end")
        self.new_pref_id.delete(0, "end")

        # success feedback
        self.feedback.configure(text=f"Added AprilTag ID {april_id} as ID {pref_id}", text_color="#66FF66")


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

        # update shared data
        self.controller.shared_data["center"] = center_id

        # update label display
        self.center_label.configure(text=f"Current center ID: {center_id}")

        # clear entry
        self.center.delete(0, "end")

        self.feedback.configure(text=f"Center ID set to {center_id}", text_color="#66FF66")

    # Called whenever this page is shown, which works well for displaying the camera
    def tkraise(self, aboveThis=None):
        self.stop_camera = False
        self.update_camera_view()
        super().tkraise(aboveThis)

    def update_camera_view(self):
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

                    preferred = self._april_to_preferred(tag.tag_id)

                    # Display tag info
                    tool_name = tool_map.get(tag.tag_id, f"Unknown Tool {tag.tag_id}")
                    cv2.putText(frame, 
                                f"{tool_name} (Preferred ID: {preferred}) pos: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}",
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

        self.after(30, self.update_camera_view)
    
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
                    available_tools=self.controller.shared_data["tool_map"],
                    storage_limit=self.controller.shared_data["storage"]
                    )
        
        self.detector = Detector(families="tag25h9")
        
        # Cutting up the screen (dimensioning)
        for i in range(14):
            self.grid_columnconfigure(i, weight=1)
        for i in range(6):
            self.grid_rowconfigure(i, weight=1)

        # Background Creation (originally for debugging, but I want to keep it tbh)
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
        self.panel = ctk.CTkFrame(self, fg_color="#A0D4FF")  # same color as before
        self.panel.grid(row=0, column=8, rowspan=3, columnspan=3, sticky="nsew", padx=40, pady=40)

        # optional: a title in the panel
        self.panel_title = ctk.CTkLabel(self.panel, text="Dashboard Panel", font=("TkDefaultFont", 16, "bold"))
        self.panel_title.pack(pady=(8, 6))

        # Output for API methods
        self.panel_body = ctk.CTkFrame(self.panel, fg_color="transparent")
        self.panel_body.pack(fill="both", expand=True, padx=10, pady=10)

        # Attached Text
        attached_text = ""
        self.attached = ctk.CTkLabel(self, fg_color="#6655FF", text=attached_text, anchor="center")
        self.attached.grid(row=0, column=11, rowspan=3, columnspan=3, sticky="nsew", padx=40, pady=40)

        # API Function Widget
        self.functions = ctk.CTkFrame(self, fg_color="#B4B4B4")
        self.functions.grid(row=3, column=8, rowspan=3, columnspan=6, sticky="nsew", padx=40, pady=40)

        # API Function Widget Construction
        api_cols = 4
        api_rows = 1
        for col in range(api_cols):
            self.functions.grid_columnconfigure(col, weight=1)
        for row in range(api_rows):
            self.functions.grid_rowconfigure(row, weight=1)

        # Widgets/Buttons within API Function Widget
        show_button = ctk.CTkButton(self.functions, text="Show All Attached Tools", fg_color="#DDDDDD", 
                                    command=lambda: self._panel_show("Seen Tags", self.view_shown_tags))
        show_button.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.visible_ids = set()

        attach_button = ctk.CTkButton(self.functions, text="Attach Tool", fg_color="#CCCCCC", 
                                      command=lambda: self._panel_show("Seen Tags", self.attach))
        attach_button.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        detach_button = ctk.CTkButton(self.functions, text="Detach Tool", fg_color="#BBBBBB", 
                                      command=lambda: self._panel_show("Seen Tags", self.detach_tool))
        detach_button.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        settings_button = ctk.CTkButton(self.functions, text="Settings Page", fg_color="#999999", 
                                        command=lambda: self.controller.show_frame("SettingsPage"))
        settings_button.grid(row=0, column=3, sticky="nsew", padx=5, pady=5)

        # Column Protect (minimum width)
        for i in [8, 9, 10, 11, 12, 13]:
            self.grid_columnconfigure(i, minsize=155)  # ensures at least 300px

        # Camera Time! (6x8 tiles covered up)
        # DISPLAY RESOLUTION: 4x3
        self.camera_label = ctk.CTkLabel(self, anchor="center")
        self.camera_label.grid(row=0, column=0, rowspan=6, columnspan=8, sticky="nsew", padx=40, pady=40)

        self.stop_camera = True  # Start stopped
        self.current_image = None

        # Bind resize event
        self.camera_label.bind("<Configure>", self.on_resize)

        # Start update loop in thread
        self.update_thread = threading.Thread(target=self.update_video, daemon=True)
        self.update_thread.start()

    def _panel_clear(self):
        for w in self.panel_body.winfo_children():
            w.destroy()

    def _panel_show(self, title: str, builder):
        self.panel_title.configure(text=title)
        self._panel_clear()
        builder(self.panel_body)  # builder represents the API function

    # Helpers (CTk-safe)
    def _tool_map(self):
        return self.controller.shared_data["tool_map"]
    
    def _get_position_id(self, april_id: int):
        """
        Given an AprilTag ID, return the preferred position ID (if mapped).
        Falls back to the raw AprilTag ID if no mapping exists.
        """
        pref_map = self.controller.shared_data.get("pos", {})
        return pref_map.get(april_id, april_id)
    
    def _april_to_preferred(self, april_id):
        # get the mapping
        pref_map = self.controller.shared_data.get("pos", {})

        # look up preferred ID if it exists, otherwise fall back to raw
        return pref_map.get(april_id, april_id)
                

    # adapter-obj Dynamic CTkFrame
    # API FUNCTIONS (add, remove, etc)

    def view_shown_tags(self, parent):
        tm = self.controller.shared_data["tool_map"]
        ids = sorted(self.visible_ids)

        title = ctk.CTkLabel(parent, text="Seen Tags (this session):", anchor="w", justify="center")
        title.pack(anchor="w", pady=(0, 6))

        if not ids:
            ctk.CTkLabel(parent, text="<no tags detected yet>", anchor="w").pack(anchor="w")
            return

        lines = [f"{tid}: {tm.get(tid, f'Unknown Tool {tid}')}" for tid in ids]
        ctk.CTkLabel(parent, text="\n".join(lines), anchor="w", justify="left").pack(anchor="w")

    def attach(self, parent):
        if len(self.attached) >= self.limit:
            error = ctk.CTkLabel(parent, text="Max limit reached.", anchor="w", justify="center")
            error.pack(anchor="w", pady=(0, 6))
            return
        
        tm = self.controller.shared_data["tool_map"]
        # tool_id assignment by number of attached already present (meaning that a tool_id can't be greater than 3)
        tool_id = len(self.attached)
        self.adapter.attached[tool_id] = {
            # "pose": {1, 2, 3, 4, 5, 6}
            # "position" : int
            # "name": "scalpel"
        }

    def detach_tool(self):
        print("detach")
    def move_tool(self):
        print("move")
    # def where(self):
    #     print("where")

    # MAIN DASHBOARD FUNCTIONS
    
    def on_resize(self, event):
        # Handle window resize and update image to match new label size
        self.label_width = event.width
        self.label_height = event.height

    def update_video(self):
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

                    # Display tag info

                    # Array of tags
                    # self.seen.append(tag)
                    tool_name = tool_map.get(tag.tag_id, f"Unknown Tool {tag.tag_id}")
                    cv2.putText(frame, 
                                f"{tool_name} (ID: {tag.tag_id}) pos: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}",
                                (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 100, 0), 
                                2)
                    
                current_seen.add(int(tag.tag_id))

            self.visible_ids = current_seen

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Wait until label is laid out
            if hasattr(self, "label_width") and hasattr(self, "label_height"):
                resized_img = self.resize_to_fit_4_3(img, self.label_width, self.label_height)
                self.imgtk = ctk.CTkImage(light_image=resized_img, size=resized_img.size)
                self.camera_label.configure(image=self.imgtk)

            # Update label image
            self.camera_label.configure(image=self.imgtk)

        # Schedule next frame update ~60 FPS
        self.camera_label.after(15, self.update_video)

    # Equation: width = height * aspect_ratio
    def resize_to_fit_4_3(self, image, max_w, max_h):
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

    def tkraise(self, aboveThis=None):

        # Start webcam update loop
        self.stop_camera = False
        self.update_video()

        super().tkraise(aboveThis)

    def on_hide(self):
        # Stop webcam loop when page is hidden
        self.stop_camera = True


class SettingsPage(BasePage):
    def __init__(self, master, controller):
        super().__init__(master, controller)
        ctk.CTkLabel(self, text="Settings Page (empty)").pack(pady=20)
        ctk.CTkButton(self, text="Settings Page", fg_color="#999999", command=lambda: self.controller.show_frame("DashboardPage")).pack(pady=20)


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

        # Load page frames into self.container
        self.frames = {}
        for PageClass in (PreparationPage, DashboardPage, SettingsPage):
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
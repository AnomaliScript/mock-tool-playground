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

        self.grid_rowconfigure(0, weight=1)
        for col in (0, 1):
            self.grid_columnconfigure(col, weight=1)

        left = ctk.CTkFrame(self, fg_color="#333333")
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        right = ctk.CTkFrame(self, fg_color="#555555")
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # ---- shared defaults
        self.controller.shared_data.setdefault("holding_limit", 6)
        self.controller.shared_data.setdefault("tool_map", {
            0: "Scalpel", 1: "Forceps", 2: "Suction Tip", 3: "Probe",
            4: "Camera Tool", 5: "Retractor", 6: "Surgical Scissors", 7: "Hemostat"
        })
        self.controller.shared_data.setdefault("center", "")

        # ---- holding limit UI
        self.holding_limit_label = ctk.CTkLabel(left, text=f"Current value: {self.controller.shared_data['holding_limit']}")
        self.holding_limit_label.pack(pady=12)
        ctk.CTkLabel(left, text="Enter how many tool slots are on the prototype").pack()
        self.hold_num = ctk.CTkEntry(left, placeholder_text="e.g. 6")
        self.hold_num.pack(pady=6)
        ctk.CTkButton(left, text="Submit New Holding Limit", command=self.submit_holding_limit).pack(pady=6)

        # ---- tool list (scrollable)
        ctk.CTkLabel(left, text="Current tools available:").pack(pady=(16, 6))
        self.tool_list_frame = ctk.CTkScrollableFrame(left, height=140)
        self.tool_list_frame.pack(fill="x", pady=(0, 10))
        self.tool_list_text = ctk.CTkLabel(self.tool_list_frame, justify="left", anchor="w")
        self.tool_list_text.pack(fill="x", padx=6, pady=6)

        # ---- add tool UI
        ctk.CTkLabel(left, text="Add a tool (Name or Name:ID)").pack(pady=(6, 4))
        self.new_tool = ctk.CTkEntry(left, placeholder_text="Retractor or Retractor:12")
        self.new_tool.pack(pady=6, fill="x")
        self.new_tool.bind("<Return>", lambda _e: self.submit_additional_tool())
        ctk.CTkButton(left, text="Submit New Tool", command=self.submit_additional_tool).pack(pady=6)

        # inline feedback label
        self.feedback = ctk.CTkLabel(left, text="", text_color="#FFCC66")
        self.feedback.pack(pady=(2, 10))

        # ---- center ID UI
        self.center_label = ctk.CTkLabel(left, text=f"Current center ID: {self.controller.shared_data['center']}")
        self.center_label.pack(pady=12)
        ctk.CTkLabel(left, text="Enter the center tag ID").pack()
        self.center = ctk.CTkEntry(left, placeholder_text="e.g. 0")
        self.center.pack(pady=6)
        ctk.CTkButton(left, text="Submit Center ID", command=self.submit_center).pack(pady=6)

        # ---- right pane camera etc. (unchanged)
        self.stop_camera = False
        self.camera_label = ctk.CTkLabel(right, text="")
        self.camera_label.pack(pady=10, expand=True)

        self.detector = Detector(families="tag25h9")

        self.login_button = ctk.CTkButton(left, text="Login", command=self.login)
        self.login_button.pack(pady=10)

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

    def _smallest_unused_id(self):
        used = set(self._tool_map().keys())
        i = 0
        while i in used:
            i += 1
        return i

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

    def _set_feedback(self, msg: str, ok: bool = True):
        color = "#A4E8A2" if ok else "#FF8A80"
        self.feedback.configure(text=msg, text_color=color)

    # ---------- button callbacks (CTk) ----------
    def submit_additional_tool(self):
        user_text = self.new_tool.get()
        name, tag_id = self._parse_tool_entry(user_text)
        if not name:
            self._set_feedback("Please enter a tool name.", ok=False)
            return

        tm = self._tool_map()

        if tag_id is None:
            tag_id = self._smallest_unused_id()
        elif tag_id in tm:
            self._set_feedback(f"ID {tag_id} already used by '{tm[tag_id]}'. Choose another.", ok=False)
            return

        tm[tag_id] = name
        self._render_tool_list()
        self._set_feedback(f"Added '{name}' as ID {tag_id}.")
        self.new_tool.delete(0, "end")

    def submit_holding_limit(self):
        val = (self.hold_num.get() or "").strip()
        if val.isdigit():
            self.controller.shared_data["holding_limit"] = int(val)
            self.holding_limit_label.configure(text=f"Current value: {val}")
            self._set_feedback("Holding limit updated.")
        else:
            self._set_feedback("Enter a whole number for holding limit.", ok=False)

    def submit_center(self):
        val = (self.center.get() or "").strip()
        if val.isdigit():
            self.controller.shared_data["center"] = int(val)
        else:
            # allow non-numeric for debugging, but tell the user
            self.controller.shared_data["center"] = val
        self.center_label.configure(text=f"Current center ID: {self.controller.shared_data['center']}")
        self._set_feedback("Center updated.")

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

                    # Display tag info
                    tool_name = tool_map.get(tag.tag_id, f"Unknown Tool {tag.tag_id}")
                    cv2.putText(frame, 
                                f"{tool_name} (ID: {tag.tag_id}) pos: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}",
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

        self.adapter = classes.ToolAdapter(available_tools=self.controller.shared_data["tool_map"], # Done
                      holding_limit=self.controller.shared_data["holding_limit"], # Done
                      possible_positions=None)
        
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
        api_cols = 5
        api_rows = 1
        for col in range(api_cols):
            self.functions.grid_columnconfigure(col, weight=1)
        for row in range(api_rows):
            self.functions.grid_rowconfigure(row, weight=1)

        # Widgets/Buttons within API Function Widget
        show_button = ctk.CTkButton(self.functions, text="Show All Attached Tools", fg_color="#DDDDDD", 
                                    command=lambda: self._panel_show("Seen Tags", self.view_shown_tags)
)
        show_button.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.visible_ids = set()

        attach_button = ctk.CTkButton(self.functions, text="Attach Tool", fg_color="#CCCCCC", 
                                      command=lambda: self.controller.attach())
        attach_button.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        detach_button = ctk.CTkButton(self.functions, text="Detach Tool", fg_color="#BBBBBB", 
                                      command=lambda: self.controller.detach())
        detach_button.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        move_button = ctk.CTkButton(self.functions, text="Move Tool", fg_color="#AAAAAA", 
                                    command=lambda: self.controller.move())
        move_button.grid(row=0, column=3, sticky="nsew", padx=5, pady=5)

        settings_button = ctk.CTkButton(self.functions, text="Settings Page", fg_color="#999999", 
                                        command=lambda: self.controller.show_frame("SettingsPage"))
        settings_button.grid(row=0, column=4, sticky="nsew", padx=5, pady=5)

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
        builder(self.panel_body)  # build content into panel_body

    # Helpers (CTk-safe)
    def _tool_map(self):
        return self.controller.shared_data["tool_map"]
    
    # adapter-obj Dynamic CTkFrame
    # API FUNCTIONS (add, remove, etc)

    def view_shown_tags(self, parent):
        tm = self.controller.shared_data["tool_map"]
        ids = sorted(self.visible_ids)

        title = ctk.CTkLabel(parent, text="Seen Tags (this session):",
                            anchor="w", justify="left")
        title.pack(anchor="w", pady=(0, 6))

        if not ids:
            ctk.CTkLabel(parent, text="<no tags detected yet>", anchor="w").pack(anchor="w")
            return

        lines = [f"{tid}: {tm.get(tid, f'Unknown Tool {tid}')}" for tid in ids]
        ctk.CTkLabel(parent, text="\n".join(lines), anchor="w", justify="left").pack(anchor="w")

    def attach(self):
        if len(self.attached) >= self.limit:
            self.adapter_obj.configure(text="Max limit reached.")
            return
        tool_id = len(self.attached)
    
    # translation zone begin
    def attach_tool(self, chosen_id, pose, target_pos, slots_obj):
        # tool_id is an integer, and pose has six coords
        # chosen_id is ths id of the tool in the self.available dict
        # Has to return a string if there is an error
        if len(self.attached) >= self.limit:
            self.adapter_obj.configure(text="Max limit reached.")
            return
        tool_id = len(self.attached)
        
        slot_id = slots_obj.find_closest_slot(pose)
        print(f"Slot ID: {slot_id}")
        print(f"{self.available}")
        self.attached[tool_id] = {
            "pose": pose,
            "slot": slot_id,
            "name": self.available[chosen_id]
        }

        # Attaching Tool Code Here
        # attach(target_pos)

        print(f"Tool '{tool_id}' attached at slot {self.attached[tool_id]["pose"]}")
        return pose
    # translation zone end

    def detach(self):
        print("detach")
    def move(self):
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
                    self.seen.append(tag)
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
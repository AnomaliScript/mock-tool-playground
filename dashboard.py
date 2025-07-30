import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

# Open webcam globally
cap = cv2.VideoCapture(0)

# Set appearance mode and theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("themes/oceanix.json")

class BasePage(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller

class LoginPage(BasePage):
    def __init__(self, master, controller):
        super().__init__(master, controller)

        ctk.CTkLabel(self, text="Enter Username").pack(pady=20)
        self.username_entry = ctk.CTkEntry(self)
        self.username_entry.pack(pady=10)
        self.login_button = ctk.CTkButton(self, text="Login", command=self.login)
        self.login_button.pack(pady=10)

    def login(self):
        username = self.username_entry.get()
        if username.strip() != "":
            self.controller.shared_data["user"] = username
            self.controller.show_frame("DashboardPage")

class DashboardPage(BasePage):
    def __init__(self, master, controller):
        super().__init__(master, controller)

        self.welcome_label = ctk.CTkLabel(self, text="")
        self.welcome_label.pack(pady=10)

        # Label to show webcam frames
        self.camera_label = ctk.CTkLabel(self)
        self.camera_label.pack(pady=50)

        self._stop_camera = True  # Start stopped
        self.current_image = None

    def update_video(self):
        if self._stop_camera:
            return

        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Convert to Tkinter PhotoImage
            imgtk = ImageTk.PhotoImage(img.resize((800, 600)), (800, 600))

            # Update label image
            self.camera_label.configure(image=imgtk)

        # Schedule next frame update ~60 FPS
        self.camera_label.after(15, self.update_video)

    def tkraise(self, aboveThis=None):

        # Start webcam update loop
        self._stop_camera = False
        self.update_video()

        super().tkraise(aboveThis)

    def on_hide(self):
        # Stop webcam loop when page is hidden
        self._stop_camera = True

class SettingsPage(BasePage):
    def __init__(self, master, controller):
        super().__init__(master, controller)
        ctk.CTkLabel(self, text="Settings Page (empty)").pack(pady=20)


# App Controller
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1280x720")

        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for PageClass in (LoginPage, DashboardPage, SettingsPage):
            page_name = PageClass.__name__
            frame = PageClass(container, self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("DashboardPage")

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
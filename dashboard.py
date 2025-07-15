import customtkinter as ctk
import random

# Appearance settings
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


# Base class for pages
class BasePage(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller


# Login Page (simple entry for demo purposes)
class LoginPage(BasePage):
    def __init__(self, master, controller):
        super().__init__(master, controller)

        ctk.CTkLabel(self, text="Username").pack(pady=10)
        self.username_entry = ctk.CTkEntry(self)
        self.username_entry.pack(pady=5)

        ctk.CTkButton(self, text="Login", command=self.login).pack(pady=20)

    def login(self):
        username = self.username_entry.get()
        if username:
            self.controller.shared_data["user"] = username
            self.controller.show_frame("DashboardPage")


# Dashboard Page (displays dynamic data)
class DashboardPage(BasePage):
    def __init__(self, master, controller):
        super().__init__(master, controller)

        self.title_label = ctk.CTkLabel(self, text="Dashboard", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=10)

        self.user_label = ctk.CTkLabel(self, text="")
        self.user_label.pack(pady=5)

        # Example data display (you can replace these with real values later)
        self.data_label_1 = ctk.CTkLabel(self, text="Temperature:")
        self.data_value_1 = ctk.CTkLabel(self, text="-- °C")
        self.data_label_1.pack()
        self.data_value_1.pack()

        self.data_label_2 = ctk.CTkLabel(self, text="Tool Attached:")
        self.data_value_2 = ctk.CTkLabel(self, text="None")
        self.data_label_2.pack()
        self.data_value_2.pack()

        ctk.CTkButton(self, text="Refresh Data", command=self.refresh_data).pack(pady=15)
        ctk.CTkButton(self, text="Logout", command=self.logout).pack()

    def tkraise(self, aboveThis=None):
        user = self.controller.shared_data.get("user", "Unknown")
        self.user_label.configure(text=f"Welcome, {user}")
        self.refresh_data()
        super().tkraise(aboveThis)

    def refresh_data(self):
        # Simulate dynamic values; replace with real data fetch logic
        temp = round(random.uniform(22, 28), 1)
        tool = random.choice(["Scalpel", "Forceps", "None"])
        self.data_value_1.configure(text=f"{temp} °C")
        self.data_value_2.configure(text=tool)

    def logout(self):
        self.controller.shared_data["user"] = None
        self.controller.show_frame("LoginPage")


# App Controller
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Surgical Tool Dashboard")
        self.geometry("400x400")
        self.shared_data = {}

        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for PageClass in (LoginPage, DashboardPage):
            name = PageClass.__name__
            frame = PageClass(container, self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LoginPage")

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()


if __name__ == "__main__":
    app = App()
    app.mainloop()
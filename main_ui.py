import customtkinter as ctk
import threading
from tkinter import filedialog
from main_function import run_bia, run_aao

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("AutoTag - PDF to TXT Converter")


def center_window(window, width=400, height=320):
    window.update_idletasks()

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))

    window.geometry(f"{width}x{height}+{x}+{y}")


def print_status(message):
    textarea.configure(state="normal")
    textarea.insert("end", message + "\n")
    textarea.see("end")
    textarea.configure(state="disabled")


def browse_callback():
    folder_path = filedialog.askdirectory(title="Select PDF Folder")

    if folder_path:
        entry.delete(0, "end")
        entry.insert(0, folder_path)
        print_status("Selected folder: " + folder_path)


def run_conversion():
    selected_type = optionmenu.get()
    folder_path = entry.get().strip()

    if not folder_path:
        print_status("Please select a PDF folder first.")
        return

    print_status("Starting conversion...")
    print_status("Selected court: " + selected_type)
    print_status("File path: " + folder_path)

    try:
        if selected_type == "AAO":
            run_aao(folder_path)
        elif selected_type == "BIA":
            run_bia(folder_path)

        print_status("Conversion complete.")

    except Exception as error:
        print_status("Error: " + str(error))


def convert_callback():
    threading.Thread(target=run_conversion, daemon=True).start()


center_window(app, 400, 320)

frame = ctk.CTkFrame(app)
frame.pack(fill="both", expand=True, padx=20, pady=20)

frame.grid_columnconfigure(0, weight=0)
frame.grid_columnconfigure(1, weight=1)
frame.grid_columnconfigure(2, weight=0)
frame.grid_rowconfigure(2, weight=1)

options = ["AAO", "BIA"]

optionmenu = ctk.CTkOptionMenu(frame, values=options, corner_radius=0, width=80)
optionmenu.grid(row=0, column=0, padx=(0, 5), pady=5)

entry = ctk.CTkEntry(frame, corner_radius=0)
entry.grid(row=0, column=1, sticky="ew", padx=(0, 5), pady=5)

button = ctk.CTkButton(
    frame,
    text="Browse",
    corner_radius=0,
    width=90,
    command=browse_callback
)
button.grid(row=0, column=2, pady=5)

btn_convert = ctk.CTkButton(
    frame,
    text="Convert",
    corner_radius=0,
    command=convert_callback
)
btn_convert.grid(row=1, column=0, columnspan=3, pady=10, sticky="ew")

textarea = ctk.CTkTextbox(frame, corner_radius=0)
textarea.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(0, 5))
textarea.configure(state="disabled")

print_status("Ready.")

app.mainloop()
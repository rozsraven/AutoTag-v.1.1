import customtkinter as ctk
import threading
from tkinter import filedialog
from main_function import run_bia, run_aao
from xml_proofer import xml_proofer_main

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("AutoTag Converter")


def center_window(window, width=700, height=420):
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

def clear_textarea():
    textarea.configure(state="normal")
    textarea.delete("1.0", "end")
    textarea.configure(state="disabled")

def browse_folder():
    folder_path = filedialog.askdirectory(title="Select Folder")

    if folder_path:
        entry.delete(0, "end")
        entry.insert(0, folder_path)
        print_status("Selected folder: " + folder_path)


def run_pdf_to_xml():
    selected_type = optionmenu.get()
    folder_path = entry.get().strip()

    if not folder_path:
        print_status("Please select a PDF folder first.")
        return

    print_status("Starting PDF to XML conversion...")
    print_status("Selected court: " + selected_type)
    print_status("File path: " + folder_path)

    try:
        if selected_type == "AAO":
            run_aao(folder_path)
        elif selected_type == "BIA":
            run_bia(folder_path)

        print_status("PDF to XML conversion complete.")

    except Exception as error:
        print_status("Error: " + str(error))


def run_xml_to_pdf():
    folder_path = entry.get().strip()

    if not folder_path:
        print_status("Please select an XML folder first.")
        return

    print_status("Starting XML to PDF conversion...")
    print_status("File path: " + folder_path)

    try:
        # Replace this with your XML to PDF function
        xml_proofer_main(folder_path)

        print_status("XML to PDF conversion complete.")

    except Exception as error:
        print_status("Error: " + str(error))


def convert_callback():
    if current_mode.get() == "PDF_TO_XML":
        threading.Thread(target=run_pdf_to_xml, daemon=True).start()
    elif current_mode.get() == "XML_TO_PDF":
        threading.Thread(target=run_xml_to_pdf, daemon=True).start()


def show_pdf_to_xml():
    current_mode.set("PDF_TO_XML")

    title_label.configure(text="PDF to XML")

    optionmenu.grid(row=0, column=0, padx=(0, 5), pady=5)
    entry.grid(row=0, column=1, sticky="ew", padx=(0, 5), pady=5)
    browse_button.grid(row=0, column=2, pady=5)

    convert_button.configure(text="Convert PDF to XML")
    print_status("Switched to PDF to XML.")


def show_xml_to_pdf():
    current_mode.set("XML_TO_PDF")

    title_label.configure(text="XML to PDF")

    optionmenu.grid_forget()
    entry.grid(row=0, column=1, sticky="ew", padx=(0, 5), pady=5)
    browse_button.grid(row=0, column=2, pady=5)

    convert_button.configure(text="Convert XML to PDF")
    print_status("Switched to XML to PDF.")

center_window(app, 700, 420)

current_mode = ctk.StringVar(value="PDF_TO_XML")

app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

sidebar = ctk.CTkFrame(app, width=160, corner_radius=0)
sidebar.grid(row=0, column=0, sticky="nsw")
sidebar.grid_propagate(False)

sidebar_title = ctk.CTkLabel(
    sidebar,
    text="AutoTag",
    font=ctk.CTkFont(size=20, weight="bold")
)
sidebar_title.pack(pady=(20, 15))

btn_pdf_to_xml = ctk.CTkButton(
    sidebar,
    text="PDF to XML",
    corner_radius=0,
    command=show_pdf_to_xml
)
btn_pdf_to_xml.pack(fill="x", padx=10, pady=5)

btn_xml_to_pdf = ctk.CTkButton(
    sidebar,
    text="XML to PDF",
    corner_radius=0,
    command=show_xml_to_pdf
)
btn_xml_to_pdf.pack(fill="x", padx=10, pady=5)

main_frame = ctk.CTkFrame(app, corner_radius=0)
main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

main_frame.grid_columnconfigure(0, weight=1)
#main_frame.grid_rowconfigure(2, weight=1)
main_frame.grid_rowconfigure(4, weight=1)

title_label = ctk.CTkLabel(
    main_frame,
    text="PDF to XML",
    font=ctk.CTkFont(size=18, weight="bold")
)
title_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

content_frame = ctk.CTkFrame(main_frame, corner_radius=0)
content_frame.grid(row=1, column=0, sticky="ew")
content_frame.grid_columnconfigure(0, weight=0)
content_frame.grid_columnconfigure(1, weight=1)
content_frame.grid_columnconfigure(2, weight=0)

options = ["AAO", "BIA"]

optionmenu = ctk.CTkOptionMenu(content_frame, values=options, corner_radius=0, width=80)
optionmenu.grid(row=0, column=0, padx=(0, 5), pady=5)

entry = ctk.CTkEntry(content_frame, corner_radius=0)
entry.grid(row=0, column=1, sticky="ew", padx=(0, 5), pady=5)

browse_button = ctk.CTkButton(
    content_frame,
    text="Browse",
    corner_radius=0,
    width=90,
    command=browse_folder
)
browse_button.grid(row=0, column=2, pady=5)

convert_button = ctk.CTkButton(
    main_frame,
    text="Convert PDF to XML",
    corner_radius=0,
    command=convert_callback
)
convert_button.grid(row=2, column=0, pady=10, sticky="ew")

# Text area top bar
textarea_topbar = ctk.CTkFrame(main_frame, corner_radius=0)
textarea_topbar.grid(row=3, column=0, sticky="ew", pady=(10, 0))

textarea_topbar.grid_columnconfigure(0, weight=1)

clear_button = ctk.CTkButton(
    textarea_topbar,
    text="Clear Logs",
    width=100,
    corner_radius=0,
    command=clear_textarea
)
clear_button.grid(row=0, column=1, sticky="e")

# Text area
textarea = ctk.CTkTextbox(main_frame, corner_radius=0)
textarea.grid(row=4, column=0, sticky="nsew", pady=(0, 5))
textarea.configure(state="disabled")

print_status("Ready.")

app.mainloop()
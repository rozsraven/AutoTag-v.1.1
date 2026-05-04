import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import traceback
import queue
import sys
import multiprocessing

from main_updated import main_function


class QueueWriter:
    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, text):
        if text:
            self.log_queue.put(text)

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Counter")
        self.root.geometry("500x350")

        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top_frame = tk.Frame(self.root)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        top_frame.columnconfigure(0, weight=1)

        self.entry = tk.Entry(top_frame, borderwidth=1, relief="solid")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.entry.bind("<Return>", lambda event: self.on_button_click())

        self.button = tk.Button(top_frame, text="Run", command=self.on_button_click)
        self.button.grid(row=0, column=1)

        self.text_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            height=10,
            borderwidth=1,
            relief="solid"
        )
        self.text_area.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.process_log_queue)

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.text_area.insert(tk.END, message)
                self.text_area.see(tk.END)
        except queue.Empty:
            pass

        self.root.after(100, self.process_log_queue)

    def log(self, message):
        self.log_queue.put(message)

    def run_main_function(self, user_input):
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = QueueWriter(self.log_queue)
            sys.stderr = QueueWriter(self.log_queue)

            print(f"> Running with input: {user_input}")
            print(f"Button clicked. User input: {user_input}")

            main_function(user_input)

            print("\nProcessing completed.\n")
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Info",
                    "Processing completed. Check the output folder for results."
                )
            )

        except Exception:
            print("\nAn error occurred:\n")
            print(traceback.format_exc())
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error",
                    "An error occurred. Check the text area for details."
                )
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.is_running = False
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.button.config(state=tk.NORMAL)
        self.entry.config(state=tk.NORMAL)
        self.entry.delete(0, tk.END)

    def on_button_click(self):
        user_input = self.entry.get().strip()

        if not user_input:
            messagebox.showwarning("Warning", "Please enter a value.")
            return

        if self.is_running:
            messagebox.showwarning("Busy", "A task is already running.")
            return

        self.is_running = True
        self.text_area.delete("1.0", tk.END)

        self.button.config(state=tk.DISABLED)
        self.entry.config(state=tk.DISABLED)

        self.worker_thread = threading.Thread(
            target=self.run_main_function,
            args=(user_input,),
            daemon=True
        )
        self.worker_thread.start()

    def on_close(self):
        if self.is_running:
            should_close = messagebox.askyesno(
                "Exit",
                "A task is still running. Close anyway?"
            )
            if not should_close:
                return

        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
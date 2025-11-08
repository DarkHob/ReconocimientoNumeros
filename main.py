# ================== PUNTO DE ENTRADA ==================
import tkinter as tk
from tkinter import ttk

from learning import train_or_load_digit_model
from ui import DigitApp

if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("clam")
    except:
        pass

    model = train_or_load_digit_model()
    app = DigitApp(root, model)
    root.mainloop()

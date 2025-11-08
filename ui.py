# ================== INTERFAZ GRÁFICA (Tkinter) ==================
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import pyttsx3

from config import CANVAS_SIZE, BRUSH_SIZE
from cameras import list_available_cameras, CameraManager
from predict import predict_number_from_gray

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

class DigitApp:
    def __init__(self, root, model):
        self.root  = root
        self.model = model
        self.root.title("Reconocimiento de Números 0–100")

        self.notebook   = ttk.Notebook(root)
        self.frame_cam  = ttk.Frame(self.notebook)
        self.frame_draw = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_cam,  text="Cámara")
        self.notebook.add(self.frame_draw, text="Dibujar")
        self.notebook.pack(fill="both", expand=True)

        # ---------- Cámara ----------
        self.cam_mgr = CameraManager()

        # Controles arriba del video
        top_controls = ttk.Frame(self.frame_cam)
        top_controls.pack(padx=10, pady=(10,0), fill="x")

        ttk.Label(top_controls, text="Cámara:").pack(side="left", padx=(0,5))
        self.cam_combo = ttk.Combobox(top_controls, state="readonly", width=10)
        self.cam_combo.pack(side="left", padx=5)
        self.refresh_btn = ttk.Button(top_controls, text="Actualizar", command=self._scan_cameras_ui)
        self.refresh_btn.pack(side="left", padx=5)
        self.connect_btn = ttk.Button(top_controls, text="Conectar", command=self._connect_selected_camera)
        self.connect_btn.pack(side="left", padx=5)

        # Espacio de video
        self.video_label = tk.Label(self.frame_cam)
        self.video_label.pack(padx=10, pady=(10,10))

        # Botón de captura debajo del video
        bottom_controls = ttk.Frame(self.frame_cam)
        bottom_controls.pack(padx=10, pady=(0,10))
        self.capture_btn = ttk.Button(bottom_controls, text="Capturar (ROI)", command=self.capture_and_predict_roi)
        self.capture_btn.pack(anchor="center")

        # Resultado
        self.result_var_cam = tk.StringVar(value="Selecciona una cámara y pulsa 'Conectar'.")
        self.result_label_cam = ttk.Label(self.frame_cam, textvariable=self.result_var_cam, font=("Arial", 14))
        self.result_label_cam.pack(pady=(0,10))

        self.running    = True
        self.last_frame = None
        self.roi_rect   = None

        self._scan_cameras_ui(auto_connect=True)
        self.update_frame()

        # ---------- Dibujo ----------
        self.canvas_size = CANVAS_SIZE
        self.brush       = BRUSH_SIZE
        self.canvas = tk.Canvas(self.frame_draw, width=self.canvas_size, height=self.canvas_size,
                                bg="white", cursor="pencil")
        self.canvas.pack(padx=10, pady=10)

        self.canvas_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.canvas_draw  = ImageDraw.Draw(self.canvas_image)

        draw_controls = ttk.Frame(self.frame_draw)
        draw_controls.pack(padx=10, pady=5, fill="x")
        self.predict_draw_btn = ttk.Button(draw_controls, text="Predecir dibujo", command=self.predict_from_canvas)
        self.predict_draw_btn.pack(side="left", padx=5)
        self.clear_btn = ttk.Button(draw_controls, text="Limpiar", command=self.clear_canvas)
        self.clear_btn.pack(side="left", padx=5)

        self.result_var_draw = tk.StringVar(value="Escribe un número (0–100) y presiona 'Predecir dibujo'")
        self.result_label_draw = ttk.Label(self.frame_draw, textvariable=self.result_var_draw, font=("Arial", 14))
        self.result_label_draw.pack(pady=5)

        # Dibujo suave
        self.last_pt = None
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- Cámara ----------
    def _scan_cameras_ui(self, auto_connect=False):
        cams = list_available_cameras(max_index=10)
        if not cams:
            self.cam_combo['values'] = []
            self.cam_combo.set("")
            self.result_var_cam.set("No se detectaron cámaras.")
            return
        self.cam_combo['values'] = [str(i) for i in cams]
        self.cam_combo.current(0)
        if auto_connect:
            self._connect_selected_camera()

    def _connect_selected_camera(self):
        sel = self.cam_combo.get()
        if not sel:
            self.result_var_cam.set("Selecciona una cámara.")
            return
        idx = int(sel)
        if not self.cam_mgr.open(idx):
            self.result_var_cam.set(f"No se pudo abrir la cámara {idx}.")
            return
        self.result_var_cam.set(f"Cámara {idx} conectada.")

    def update_frame(self):
        if not self.running:
            return
        if self.cam_mgr.is_open():
            ret, frame = self.cam_mgr.read()
            if ret:
                self.last_frame = frame.copy()
                disp = frame.copy()

                # ROI centrado
                h, w = disp.shape[:2]
                side = int(min(w, h) * 0.6)
                x1 = (w - side)//2
                y1 = (h - side)//2
                x2 = x1 + side
                y2 = y1 + side
                self.roi_rect = (x1, y1, x2, y2)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            else:
                self.video_label.configure(text="No se pudo leer de la cámara.")
        else:
            self.video_label.configure(text="Sin cámara conectada.")
        self.root.after(15, self.update_frame)

    def capture_and_predict_roi(self):
        if not self.cam_mgr.is_open():
            self.result_var_cam.set("Sin cámara activa.")
            return
        if self.last_frame is None or self.roi_rect is None:
            self.result_var_cam.set("No hay imagen disponible.")
            return
        x1, y1, x2, y2 = self.roi_rect
        crop_bgr = self.last_frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        value, detail = predict_number_from_gray(gray, self.model)
        if value is None:
            self.result_var_cam.set(detail)
        else:
            self.result_var_cam.set(detail)
            speak(f"Es el número {value}")

    # ---------- Dibujo ----------
    def _start_stroke(self, event):
        self.last_pt = (event.x, event.y)

    def _draw_stroke(self, event):
        if self.last_pt is None:
            self.last_pt = (event.x, event.y)
        x0, y0 = self.last_pt
        x1, y1 = event.x, event.y
        self.canvas.create_line(x0, y0, x1, y1,
                                width=self.brush, fill="black",
                                capstyle=tk.ROUND, smooth=True)
        self.canvas_draw.line([x0, y0, x1, y1],
                              fill="black", width=self.brush, joint="curve")
        self.last_pt = (x1, y1)

    def _end_stroke(self, event):
        self.last_pt = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas_draw.rectangle([0,0,self.canvas_size,self.canvas_size], fill="white")
        self.result_var_draw.set("Lienzo limpio. Escribe un número (0–100) y presiona 'Predecir dibujo'.")

    def predict_from_canvas(self):
        img = self.canvas_image.convert("L")
        gray = np.array(img)
        value, detail = predict_number_from_gray(gray, self.model)
        if value is None:
            self.result_var_draw.set(detail)
        else:
            self.result_var_draw.set(detail)
            speak(f"Es el número {value}")

    # ---------- Cierre ----------
    def on_close(self):
        self.running = False
        try:
            self.cam_mgr.close()
        except:
            pass
        self.root.destroy()

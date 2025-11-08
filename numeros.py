from __future__ import absolute_import, division, print_function, unicode_literals
import os, math
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw

import pyttsx3
import tensorflow as tf
import tensorflow_datasets as tfds

# ================== CONFIG ==================
MODEL_PATH = "mnist_digit_cnn.keras"  # modelo de dígitos 0-9
BATCHSIZE  = 64
EPOCHS     = 5
IMG_SIZE   = 28
CLASS_DIGITS = [str(i) for i in range(10)]  # 0..9

# ================== HELPER: LISTAR CÁMARAS ==================
def list_available_cameras(max_index=10):
    """
    Devuelve una lista de índices de cámaras disponibles [0, 1, 2, ...].
    Prueba a abrir cada índice; si abre y lee un frame, se considera válida.
    """
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            if ret:
                found.append(i)
        cap.release()
    return found

# ================== MODELO (CNN dígitos 0-9, entrenado con MNIST) ==================
def build_digit_model():
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')  # 0..9
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_or_load_digit_model():
    if os.path.exists(MODEL_PATH):
        print(f"Cargando modelo de dígitos desde {MODEL_PATH} ...")
        return tf.keras.models.load_model(MODEL_PATH)

    print("No hay modelo. Entrenando con MNIST (0-9) ...")
    (ds, info) = tfds.load('mnist', as_supervised=True, with_info=True)
    train_ds, test_ds = ds['train'], ds['test']

    def normalize(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, -1)  # (28,28,1)
        return img, label

    train_ds = train_ds.map(normalize).shuffle(10000).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.map(normalize).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

    model = build_digit_model()
    model.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=test_ds)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Accuracy en test MNIST: {test_acc:.4f}")
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    return model

# ================== PREPROCESADO ESTILO MNIST ==================
def _mnist_like_from_gray(gray):
    inv = 255 - gray
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        return np.expand_dims(canvas.astype(np.float32)/255.0, -1)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = inv[y1:y2+1, x1:x2+1]
    h, w = crop.shape
    if h > w:
        new_h, new_w = 20, max(1, int(round(w * 20.0 / h)))
    else:
        new_w, new_h = 20, max(1, int(round(h * 20.0 / w)))
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y_off = (IMG_SIZE - crop.shape[0]) // 2
    x_off = (IMG_SIZE - crop.shape[1]) // 2
    canvas[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = crop
    ys_grid, xs_grid = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    m = canvas.astype(np.float32)
    msum = m.sum()
    if msum > 0:
        cy = (m * ys_grid).sum() / msum
        cx = (m * xs_grid).sum() / msum
        shift_x = int(round(IMG_SIZE//2 - cx))
        shift_y = int(round(IMG_SIZE//2 - cy))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    arr = (canvas.astype(np.float32) / 255.0)
    arr = np.expand_dims(arr, axis=-1)
    return arr

# ================== PREDICCIÓN ==================
def _predict_digit_with_tta(model, gray_crop):
    shifts = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
    probs = []
    for dx,dy in shifts:
        arr = _mnist_like_from_gray(gray_crop)[...,0]
        M = np.float32([[1,0,dx],[0,1,dy]])
        shifted = cv2.warpAffine((arr*255).astype(np.uint8), M, (IMG_SIZE,IMG_SIZE),
                                 flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)/255.0
        x = np.expand_dims(shifted, axis=(0,-1))
        p = model.predict(x, verbose=0)[0]
        probs.append(p)
    probs = np.mean(probs, axis=0)
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs

def segment_digits_from_gray(gray, max_digits=3):
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(255 - blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = gray.shape[:2]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 80 or w < 3 or h < 6:
            continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: b[0])
    crops = []
    for (x,y,w,h) in boxes[:max_digits]:
        pad = max(2, int(0.15*max(w,h)))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad); y2 = min(gray.shape[0], y + h + pad)
        crops.append(gray[y1:y2, x1:x2])
    return crops

def predict_number_from_gray(gray, model):
    digit_crops = segment_digits_from_gray(gray, max_digits=3)
    if not digit_crops:
        return None, "No se detectaron dígitos."
    preds_str, confs = [], []
    for crop in digit_crops:
        idx, conf, _ = _predict_digit_with_tta(model, crop)
        preds_str.append(CLASS_DIGITS[idx])
        confs.append(conf)
    number_str = "".join(preds_str)
    try:
        value = int(number_str.lstrip('0') or '0')
    except:
        return None, f"Predicción no válida: {number_str}"
    if value > 100:
        return None, f"{number_str} fuera de rango (0–100)."
    detail = " + ".join([f"{d} ({c:.2f})" for d,c in zip(preds_str, confs)]) + f"  →  Número: {value}"
    return value, detail

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# ================== APP ==================
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Números 0–100")
        self.model = train_or_load_digit_model()

        self.notebook = ttk.Notebook(root)
        self.frame_cam = ttk.Frame(self.notebook)
        self.frame_draw = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_cam, text="Cámara")
        self.notebook.add(self.frame_draw, text="Dibujar")
        self.notebook.pack(fill="both", expand=True)

        # ---------- Cámara ----------
        self.cap = None

        # Controles arriba del video
        top_controls = ttk.Frame(self.frame_cam)
        top_controls.pack(padx=10, pady=(10,0), fill="x")

        ttk.Label(top_controls, text="Cámara:").pack(side="left", padx=(0,5))
        self.cam_combo = ttk.Combobox(top_controls, state="readonly", width=10)
        self.cam_combo.pack(side="left", padx=5)
        self.refresh_cams_btn = ttk.Button(top_controls, text="Actualizar", command=self._scan_cameras_ui)
        self.refresh_cams_btn.pack(side="left", padx=5)
        self.connect_cam_btn = ttk.Button(top_controls, text="Conectar", command=self._connect_selected_camera)
        self.connect_cam_btn.pack(side="left", padx=5)

        # Espacio del video
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

        self.running = True
        self.last_frame = None
        self.roi_rect = None
        self._scan_cameras_ui(auto_connect=True)
        self.update_frame()

        # ---------- Dibujo ----------
        self.canvas_size = 360
        self.brush = 20
        self.canvas = tk.Canvas(self.frame_draw, width=self.canvas_size, height=self.canvas_size, bg="white", cursor="pencil")
        self.canvas.pack(padx=10, pady=10)

        self.canvas_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)

        draw_controls = ttk.Frame(self.frame_draw)
        draw_controls.pack(padx=10, pady=5, fill="x")
        self.predict_draw_btn = ttk.Button(draw_controls, text="Predecir dibujo", command=self.predict_from_canvas)
        self.predict_draw_btn.pack(side="left", padx=5)
        self.clear_btn = ttk.Button(draw_controls, text="Limpiar", command=self.clear_canvas)
        self.clear_btn.pack(side="left", padx=5)

        self.result_var_draw = tk.StringVar(value="Escribe un número (0–100) y presiona 'Predecir dibujo'")
        self.result_label_draw = ttk.Label(self.frame_draw, textvariable=self.result_var_draw, font=("Arial", 14))
        self.result_label_draw.pack(pady=5)

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
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.result_var_cam.set(f"No se pudo abrir la cámara {idx}.")
            self.cap = None
            return
        self.result_var_cam.set(f"Cámara {idx} conectada.")

    def update_frame(self):
        if not self.running:
            return
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                disp = frame.copy()
                h, w = disp.shape[:2]
                side = int(min(w, h) * 0.6)
                x1 = (w - side)//2
                y1 = (h - side)//2
                x2 = x1 + side
                y2 = y1 + side
                self.roi_rect = (x1, y1, x2, y2)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        self.root.after(15, self.update_frame)

    def capture_and_predict_roi(self):
        if self.cap is None or not self.cap.isOpened():
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
    def _start_stroke(self, event): self.last_pt = (event.x, event.y)
    def _draw_stroke(self, event):
        if self.last_pt:
            x0, y0 = self.last_pt
            self.canvas.create_line(x0, y0, event.x, event.y,
                                    width=self.brush, fill="black",
                                    capstyle=tk.ROUND, smooth=True)
            self.canvas_draw.line([x0, y0, event.x, event.y],
                                  fill="black", width=self.brush)
            self.last_pt = (event.x, event.y)
    def _end_stroke(self, event): self.last_pt = None
    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas_draw.rectangle([0,0,self.canvas_size,self.canvas_size], fill="white")
        self.result_var_draw.set("Lienzo limpio.")
    def predict_from_canvas(self):
        gray = np.array(self.canvas_image.convert("L"))
        value, detail = predict_number_from_gray(gray, self.model)
        if value is None: self.result_var_draw.set(detail)
        else:
            self.result_var_draw.set(detail)
            speak(f"Es el número {value}")

    # ---------- Cierre ----------
    def on_close(self):
        self.running = False
        try:
            if self.cap: self.cap.release()
        except: pass
        self.root.destroy()

# ================== MAIN ==================
if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("clam")
    except:
        pass
    app = DigitApp(root)
    root.mainloop()

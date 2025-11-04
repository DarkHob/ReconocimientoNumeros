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
    """Convierte un recorte gris (0-255, trazo oscuro) a (28,28,1) trazo claro centrado."""
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

# ================== UTIL: PROYECCIÓN Y SPLITS ==================
def _vertical_projection_split(mask, parts_hint=None):
    """
    Divide un bloque ancho usando proyección vertical (suma por columnas).
    Devuelve lista de (x1,x2) cortes. Si no encuentra valles claros, devuelve [0, W].
    """
    W = mask.shape[1]
    colsum = mask.sum(axis=0).astype(np.float32)
    # normalizar
    if colsum.max() > 0:
        colsum /= colsum.max()

    # suavizar señal para valles más estables
    k = max(3, W // 20 | 1)  # kernel impar
    colsum_smooth = cv2.GaussianBlur(colsum.reshape(1,-1), (k,1), 0).ravel()

    # umbral para valle (por debajo de 0.25 del pico)
    thresh = 0.25
    valleys = np.where(colsum_smooth < thresh)[0]

    if len(valleys) == 0:
        return [(0, W)]

    # agrupar valles contiguos y tomar centro de cada grupo
    splits = []
    start = valleys[0]
    for i in range(1, len(valleys)):
        if valleys[i] != valleys[i-1] + 1:
            mid = (start + valleys[i-1]) // 2
            splits.append(mid)
            start = valleys[i]
    mid = (start + valleys[-1]) // 2
    splits.append(mid)

    # limitar a 1 o 2 cortes (→ 2 o 3 dígitos)
    if parts_hint == 2 and len(splits) > 1:
        # tomar el valle más profundo
        depth = colsum_smooth[splits]
        splits = [int(splits[np.argmin(depth)])]
    elif parts_hint == 3 and len(splits) >= 2:
        # tomar 2 valles más profundos
        depth = colsum_smooth[splits]
        idx = np.argsort(depth)[:2]
        splits = sorted([int(splits[i]) for i in idx])
    else:
        # si no hay hint, máximo 2 cortes
        if len(splits) > 2:
            depth = colsum_smooth[splits]
            idx = np.argsort(depth)[:2]
            splits = sorted([int(splits[i]) for i in idx])

    # construir rangos a partir de splits
    ranges = []
    last = 0
    for s in splits:
        if s - last > 3:
            ranges.append((last, s))
        last = s
    if W - last > 3:
        ranges.append((last, W))
    if not ranges:
        ranges = [(0, W)]
    return ranges

def _split_wide_box(gray, box, max_parts=3):
    """
    Si una caja es ancha (varios dígitos pegados), intenta dividirla en 2-3 partes.
    """
    x,y,w,h = box
    crop = gray[y:y+h, x:x+w]
    # umbral e inversión para proyección
    _, bw = cv2.threshold(255 - crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ranges = _vertical_projection_split(bw, parts_hint=None)
    # limitar nr partes
    if len(ranges) > max_parts:
        ranges = ranges[:max_parts]
    new_boxes = []
    for (x1,x2) in ranges:
        ww = x2 - x1
        if ww < 4: 
            continue
        new_boxes.append((x + x1, y, ww, h))
    return new_boxes if new_boxes else [box]

# ================== SEGMENTACIÓN DE DÍGITOS ==================
def segment_digits_from_gray(gray, max_digits=3):
    """
    Imagen gris (fondo blanco, trazo negro) -> lista de recortes por dígito ordenados izq→der.
    Usa dilatación leve + contornos; divide automáticamente bloques anchos por proyección vertical.
    """
    # Suavizar y binarizar
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(255 - blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Unir trazos cercanos para evitar cortes raros
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.dilate(bw, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = gray.shape[:2]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 80 or w < 3 or h < 6:
            continue
        # descartar marcos/extremos
        if x <= 1 or y <= 1 or x+w >= W-1 or y+h >= H-1:
            pass
        boxes.append((x,y,w,h))

    if not boxes:
        return []

    # Orden inicial por X
    boxes.sort(key=lambda b: b[0])

    # Intento de split en cajas demasiado anchas
    refined = []
    for b in boxes:
        x,y,w,h = b
        aspect = w / float(max(h,1))
        if aspect > 0.9*max_digits:  # muy ancho para un dígito
            refined.extend(_split_wide_box(gray, b, max_parts=max_digits))
        elif aspect > 1.7:  # heurística: probablemente 2 dígitos
            refined.extend(_split_wide_box(gray, b, max_parts=2))
        else:
            refined.append(b)

    # Si aún queda una sola caja muy ancha y esperamos múltiples dígitos, dividirla
    if len(refined) == 1 and max_digits > 1:
        x,y,w,h = refined[0]
        aspect = w / float(max(h,1))
        if aspect > 1.6:
            refined = _split_wide_box(gray, refined[0], max_parts=max_digits)

    # Orden final por X y recortes con padding
    refined.sort(key=lambda b: b[0])
    crops = []
    for (x,y,w,h) in refined[:max_digits]:
        pad = max(2, int(0.15*max(w,h)))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad); y2 = min(gray.shape[0], y + h + pad)
        crops.append(gray[y1:y2, x1:x2])
    return crops

# ================== PREDICCIÓN CON TTA ==================
def _predict_digit_with_tta(model, gray_crop):
    """
    TTA: pequeñas traslaciones (-1,0,+1) en x/y. Se promedia la probabilidad.
    """
    shifts = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
    probs = []
    for dx,dy in shifts:
        # trasladar el recorte suavemente en canvas 28x28
        arr = _mnist_like_from_gray(gray_crop)[...,0]  # (28,28)
        M = np.float32([[1,0,dx],[0,1,dy]])
        shifted = cv2.warpAffine((arr*255).astype(np.uint8), M, (IMG_SIZE,IMG_SIZE),
                                 flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)/255.0
        x = np.expand_dims(shifted, axis=(0,-1))  # (1,28,28,1)
        p = model.predict(x, verbose=0)[0]
        probs.append(p)
    probs = np.mean(probs, axis=0)
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs

def predict_number_from_gray(gray, model):
    """
    Segmenta 1–3 dígitos, predice cada uno con TTA y concatena.
    Devuelve (numero_predicho:int o None, detalle_texto).
    """
    digit_crops = segment_digits_from_gray(gray, max_digits=3)
    if not digit_crops:
        return None, "No se detectaron dígitos."

    preds_str, confs = [], []
    for crop in digit_crops:
        idx, conf, _ = _predict_digit_with_tta(model, crop)
        preds_str.append(CLASS_DIGITS[idx])
        confs.append(conf)

    number_str = "".join(preds_str)
    # normalizar 0..100 (quitar ceros a la izquierda)
    try:
        value = int(number_str.lstrip('0') or '0')
    except:
        return None, f"Predicción no válida: {number_str}"

    if value > 100:
        # Si se pasó, intenta dividir en más partes (ya limitado a 3) o reporta
        return None, f"{number_str} fuera de rango (0–100)."

    detail = " + ".join([f"{d} ({c:.2f})" for d,c in zip(preds_str, confs)]) + f"  →  Número: {value}"
    return value, detail

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS] No se pudo reproducir voz: {e}")

# ================== APP ==================
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Números 0–100 (segmentación + TTA)")
        self.model = train_or_load_digit_model()

        self.notebook = ttk.Notebook(root)
        self.frame_cam = ttk.Frame(self.notebook)
        self.frame_draw = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_cam, text="Cámara")
        self.notebook.add(self.frame_draw, text="Dibujar")
        self.notebook.pack(fill="both", expand=True)

        # ---------- Cámara ----------
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara (VideoCapture(0)).")
        self.video_label = tk.Label(self.frame_cam)
        self.video_label.pack(padx=10, pady=10)

        cam_controls = ttk.Frame(self.frame_cam)
        cam_controls.pack(padx=10, pady=5, fill="x")
        self.capture_btn = ttk.Button(cam_controls, text="Capturar (ROI)", command=self.capture_and_predict_roi)
        self.capture_btn.pack(side="left", padx=5)

        self.result_var_cam = tk.StringVar(value="Esperando captura...")
        self.result_label_cam = ttk.Label(self.frame_cam, textvariable=self.result_var_cam, font=("Arial", 14))
        self.result_label_cam.pack(pady=5)

        self.running = True
        self.last_frame = None
        self.roi_rect = None
        self.update_frame()

        # ---------- Lienzo ----------
        self.canvas_size = 280
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

        # Dibujo suave
        self.last_pt = None
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- Cámara ----------
    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
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
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_frame)

    def capture_and_predict_roi(self):
        if self.last_frame is None or self.roi_rect is None:
            self.result_var_cam.set("No hay frame disponible aún.")
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
            self.cap.release()
        except:
            pass
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

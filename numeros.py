from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw

import pyttsx3
import tensorflow as tf
import tensorflow_datasets as tfds

MODEL_PATH = "mnist_mlp.keras"
BATCHSIZE = 32
EPOCHS = 5
CLASS_NAMES = [str(i) for i in range(10)]

# ================== MODELO ==================

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        print(f"Cargando modelo desde {MODEL_PATH} ...")
        return tf.keras.models.load_model(MODEL_PATH)

    print("No se encontró modelo guardado. Entrenando uno nuevo (MNIST)...")
    (ds, meta) = tfds.load('mnist', as_supervised=True, with_info=True)
    train_ds, test_ds = ds['train'], ds['test']

    def normalize(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    train_ds = train_ds.map(normalize).shuffle(10000).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.map(normalize).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.fit(train_ds, epochs=EPOCHS, verbose=1)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Accuracy en test: {test_acc:.4f}")
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    return model

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS] No se pudo reproducir voz: {e}")

# ================== PREPROCESADO ==================

def preprocess_digit_bgr(frame_bgr):
    """
    Prepara una captura BGR (OpenCV) a formato MNIST: (28,28,1) float32 en [0,1].
    Busca contorno principal, centra, hace cuadrado, reescala y ajusta contraste/inversión.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(blur, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        digit = gray[y:y+h, x:x+w] if (w*h > 300) else gray
    else:
        digit = gray

    # cuadrado con padding blanco
    h, w = digit.shape
    side = max(h, w)
    square = np.full((side, side), 255, dtype=np.uint8)
    y_off = (side - h)//2
    x_off = (side - w)//2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    resized = cv2.resize(square, (28,28), interpolation=cv2.INTER_AREA)

    # Inversión: MNIST tiene dígito claro sobre fondo oscuro.
    # Si la media es alta, invertimos.
    if resized.mean() > 127:
        resized = 255 - resized

    img = resized.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def preprocess_digit_pil(pil_img):
    """
    Toma una imagen PIL (del lienzo de dibujo), la convierte a formato MNIST.
    Asumimos lienzo blanco y trazos negros.
    """
    img = pil_img.convert("L")  # gris
    # invertir a fondo oscuro, trazo claro
    inv = Image.eval(img, lambda p: 255 - p)
    inv = inv.resize((28,28), Image.Resampling.LANCZOS)
    arr = np.array(inv).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr

# ================== APP ==================

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Dígitos - Cámara y Dibujo")
        self.model = train_or_load_model()

        self.notebook = ttk.Notebook(root)
        self.frame_cam = ttk.Frame(self.notebook)
        self.frame_draw = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_cam, text="Cámara")
        self.notebook.add(self.frame_draw, text="Dibujar")
        self.notebook.pack(fill="both", expand=True)

        # ---------- Cámara + ROI ----------
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
        self.roi_rect = None  # (x1,y1,x2,y2) en coordenadas del frame
        self.update_frame()

        # ---------- Lienzo de dibujo ----------
        self.canvas_size = 280  # 10x MNIST para facilidad
        self.brush = 16         # grosor del trazo
        self.canvas = tk.Canvas(self.frame_draw, width=self.canvas_size, height=self.canvas_size, bg="white", cursor="pencil")
        self.canvas.pack(padx=10, pady=10)

        # Imagen PIL donde “pintamos” para poder exportar
        self.canvas_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)

        draw_controls = ttk.Frame(self.frame_draw)
        draw_controls.pack(padx=10, pady=5, fill="x")

        self.predict_draw_btn = ttk.Button(draw_controls, text="Predecir dibujo", command=self.predict_from_canvas)
        self.predict_draw_btn.pack(side="left", padx=5)

        self.clear_btn = ttk.Button(draw_controls, text="Limpiar", command=self.clear_canvas)
        self.clear_btn.pack(side="left", padx=5)

        self.result_var_draw = tk.StringVar(value="Dibuja un número y presiona 'Predecir dibujo'")
        self.result_label_draw = ttk.Label(self.frame_draw, textvariable=self.result_var_draw, font=("Arial", 14))
        self.result_label_draw.pack(pady=5)

        # Eventos de dibujo (mouse o táctil que emula mouse)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- Cámara ----------
    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()
            disp = frame.copy()

            # definimos ROI centrado (cuadrado) según tamaño del frame
            h, w = disp.shape[:2]
            side = int(min(w, h) * 0.6)  # 60% del lado menor
            x1 = (w - side)//2
            y1 = (h - side)//2
            x2 = x1 + side
            y2 = y1 + side
            self.roi_rect = (x1, y1, x2, y2)

            # dibujar recuadro guía (verde)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # convertir a RGB para Tkinter
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
        crop = self.last_frame[y1:y2, x1:x2]
        img28 = preprocess_digit_bgr(crop)
        x = np.expand_dims(img28, axis=0)
        preds = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        prob = float(preds[pred_idx])
        msg = f"Es el número {CLASS_NAMES[pred_idx]} (prob: {prob:.2f})"
        self.result_var_cam.set(msg)
        speak(f"Es el número {CLASS_NAMES[pred_idx]}")

    # ---------- Dibujo ----------
    def paint(self, event):
        r = self.brush // 2
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        # dibuja en el canvas visible
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        # dibuja en la imagen PIL (para preprocesado)
        self.canvas_draw.ellipse([x1, y1, x2, y2], fill="black", outline="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas_draw.rectangle([0,0,self.canvas_size,self.canvas_size], fill="white")
        self.result_var_draw.set("Lienzo limpio. Dibuja un número y presiona 'Predecir dibujo'.")

    def predict_from_canvas(self):
        img28 = preprocess_digit_pil(self.canvas_image)
        x = np.expand_dims(img28, axis=0)
        preds = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        prob = float(preds[pred_idx])
        msg = f"Es el número {CLASS_NAMES[pred_idx]} (prob: {prob:.2f})"
        self.result_var_draw.set(msg)
        speak(f"Es el número {CLASS_NAMES[pred_idx]}")

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

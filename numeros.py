from __future__ import annotations
import os
import sys
import numpy as np
import cv2
import pyttsx3
import tensorflow as tf
import tensorflow_datasets as tfds

from PyQt6 import QtCore, QtGui, QtWidgets

MODEL_PATH = "mnist_cnn.keras"
BATCHSIZE = 32
EPOCHS = 10
CLASS_NAMES = [str(i) for i in range(10)]

# ================== MODELO ==================

def build_model():
    # Modelo CNN: muchas más conexiones (pesos) que una MLP simple
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
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
    # EarlyStopping para evitar sobreentrenamiento y ModelCheckpoint para guardar el mejor
    es = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor='val_accuracy')
    ck = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')
    model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks=[es, ck], verbose=1)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Accuracy en test: {test_acc:.4f}")
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    return model

def speak(text: str):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS] No se pudo reproducir voz: {e}")

# ================== PREPROCESADO ==================

def preprocess_digit_bgr(frame_bgr: np.ndarray) -> np.ndarray:
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

    h, w = digit.shape
    side = max(h, w)
    square = np.full((side, side), 255, dtype=np.uint8)
    y_off = (side - h)//2
    x_off = (side - w)//2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    resized = cv2.resize(square, (28,28), interpolation=cv2.INTER_AREA)

    # Si la media es alta, invertimos (MNIST: dígito claro sobre fondo oscuro)
    if resized.mean() > 127:
        resized = 255 - resized

    img = resized.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

# ================== WIDGETS PERSONALIZADOS ==================

class PaintCanvas(QtWidgets.QLabel):
    def __init__(self, size=280, brush=18, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.pixmap_obj = QtGui.QPixmap(size, size)
        self.pixmap_obj.fill(QtGui.QColor('white'))
        self.setPixmap(self.pixmap_obj)
        self.last_pos = None
        self.brush = brush
        self.pen_color = QtGui.QColor('black')
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.last_pos = e.position().toPoint()
        self._draw_point(self.last_pos)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.last_pos is None:
            return
        current = e.position().toPoint()
        painter = QtGui.QPainter(self.pixmap_obj)
        pen = QtGui.QPen(self.pen_color, self.brush, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(self.last_pos, current)
        painter.end()
        self.last_pos = current
        self.setPixmap(self.pixmap_obj)
        self.update()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self.last_pos = None

    def _draw_point(self, pos: QtCore.QPoint):
        painter = QtGui.QPainter(self.pixmap_obj)
        pen = QtGui.QPen(self.pen_color, self.brush, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPoint(pos)
        painter.end()
        self.setPixmap(self.pixmap_obj)
        self.update()

    def clear(self):
        self.pixmap_obj.fill(QtGui.QColor('white'))
        self.setPixmap(self.pixmap_obj)
        self.update()

    def to_mnist_array(self) -> np.ndarray:
        # Convertir a imagen en escala de grises y a numpy, respetando bytesPerLine
        qimg = self.pixmap_obj.toImage().convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
        h = qimg.height()
        w = qimg.width()
        bpl = qimg.bytesPerLine()
        ptr = qimg.bits()
        ptr.setsize(h * bpl)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, bpl))[:, :w]
        # Invertir para MNIST (dígito claro sobre fondo oscuro)
        arr = 255 - arr
        arr = cv2.resize(arr, (28,28), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
        return arr

# ================== TABS ==================

class CameraTab(QtWidgets.QWidget):
    def __init__(self, model: tf.keras.Model, parent=None):
        super().__init__(parent)
        self.model = model
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_label = QtWidgets.QLabel("Coloca el dígito dentro del recuadro y presiona Capturar")
        self.result_label.setProperty("class", "lead")

        self.capture_btn = QtWidgets.QPushButton("Capturar (ROI)")
        self.capture_btn.setProperty("variant", "primary")
        self.capture_btn.clicked.connect(self.capture_and_predict)

        layout = QtWidgets.QVBoxLayout(self)
        card = self._make_card([self.video_label])
        layout.addWidget(card)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.capture_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addWidget(self.result_label)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara (VideoCapture(0)).")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)

        self.last_frame = None
        self.roi_rect = None  # x1,y1,x2,y2 en coords del frame

    def _make_card(self, widgets: list[QtWidgets.QWidget]) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setProperty("class", "card")
        v = QtWidgets.QVBoxLayout(card)
        for w in widgets:
            v.addWidget(w)
        return card

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.last_frame = frame.copy()
        disp = frame.copy()
        h, w = disp.shape[:2]
        side = int(min(w, h) * 0.6)
        x1 = (w - side)//2
        y1 = (h - side)//2
        x2 = x1 + side
        y2 = y1 + side
        self.roi_rect = (x1, y1, x2, y2)
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)

        # Convertir a QImage y mostrar
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    def capture_and_predict(self):
        if self.last_frame is None or self.roi_rect is None:
            self.result_label.setText("No hay frame disponible aún.")
            return
        x1, y1, x2, y2 = self.roi_rect
        crop = self.last_frame[y1:y2, x1:x2]
        img28 = preprocess_digit_bgr(crop)
        x = np.expand_dims(img28, axis=0)
        preds = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        prob = float(preds[pred_idx])
        msg = f"Es el número {CLASS_NAMES[pred_idx]} (prob: {prob:.2f})"
        self.result_label.setText(msg)
        speak(f"Es el número {CLASS_NAMES[pred_idx]}")

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass

class DrawTab(QtWidgets.QWidget):
    def __init__(self, model: tf.keras.Model, parent=None):
        super().__init__(parent)
        self.model = model
        self.canvas = PaintCanvas(size=280, brush=20)
        self.predict_btn = QtWidgets.QPushButton("Predecir dibujo")
        self.predict_btn.setProperty("variant", "primary")
        self.predict_btn.clicked.connect(self.predict_from_canvas)

        self.clear_btn = QtWidgets.QPushButton("Limpiar")
        self.clear_btn.setProperty("variant", "secondary")
        self.clear_btn.clicked.connect(self.canvas.clear)

        self.result_label = QtWidgets.QLabel("Dibuja un número y presiona Predecir")
        self.result_label.setProperty("class", "lead")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._make_card([self.canvas]))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.predict_btn)
        row.addWidget(self.clear_btn)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.result_label)

    def _make_card(self, widgets: list[QtWidgets.QWidget]) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setProperty("class", "card")
        v = QtWidgets.QVBoxLayout(card)
        for w in widgets:
            v.addWidget(w)
        return card

    def predict_from_canvas(self):
        img28 = self.canvas.to_mnist_array()
        x = np.expand_dims(img28, axis=0)
        preds = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        prob = float(preds[pred_idx])
        msg = f"Es el número {CLASS_NAMES[pred_idx]} (prob: {prob:.2f})"
        self.result_label.setText(msg)
        speak(f"Es el número {CLASS_NAMES[pred_idx]}")

# ================== MAIN WINDOW ==================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocimiento de Dígitos - PyQt6 (estilo Bootstrap)")
        self.resize(900, 700)

        # Cargar/entrenar modelo
        self.model = train_or_load_model()

        # Tabs
        tabs = QtWidgets.QTabWidget()
        self.cam_tab = CameraTab(self.model)
        self.draw_tab = DrawTab(self.model)
        tabs.addTab(self.cam_tab, "Cámara")
        tabs.addTab(self.draw_tab, "Dibujar")

        # Contenedor central con padding (tipo container)
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addWidget(tabs)
        self.setCentralWidget(central)

        # Estilo tipo Bootstrap con QSS
        self.setStyleSheet(self.bootstrap_like_qss())

    def bootstrap_like_qss(self) -> str:
        return """
        QWidget { font-family: 'Segoe UI', 'Inter', Arial, sans-serif; font-size: 14px; color: #212529; }
        QMainWindow { background: #f8f9fa; }
        QTabWidget::pane { border: 1px solid #dee2e6; border-radius: 8px; background: #ffffff; }
        QTabBar::tab { background: #e9ecef; border: 1px solid #dee2e6; padding: 8px 16px; margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px; color: #212529; }
        QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; }

        .card { background: #ffffff; border: 1px solid #dee2e6; border-radius: 12px; padding: 12px; }
        QLabel.lead { color: #495057; font-size: 16px; }

        QPushButton { border-radius: 10px; padding: 8px 14px; border: 1px solid transparent; background: #e9ecef; color: #212529; }
        QPushButton:hover { filter: brightness(1.03); }
        QPushButton:pressed { padding-top: 9px; padding-bottom: 7px; }

        /* Variantes tipo Bootstrap usando dynamic property 'variant' */
        QPushButton[variant="primary"] { background: #0d6efd; color: #ffffff; }
        QPushButton[variant="primary"]:hover { background: #0b5ed7; }
        QPushButton[variant="secondary"] { background: #6c757d; color: #ffffff; }
        QPushButton[variant="secondary"]:hover { background: #5c636a; }
        """

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.cam_tab.timer.stop()
            self.cam_tab.close()
        except Exception:
            pass
        return super().closeEvent(event)

# ================== ENTRYPOINT ==================

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

# ================== VISIÓN: PREPROCESADO Y SEGMENTACIÓN ==================
import numpy as np
import cv2
from config import IMG_SIZE

def mnist_like_from_gray(gray):
    """
    Convierte recorte (gris, trazo oscuro) a (IMG_SIZE,IMG_SIZE,1) con trazo claro y centrado.
    """
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

def segment_digits_from_gray(gray, max_digits=3):
    """
    Imagen gris (fondo blanco, trazo negro) -> lista de recortes por dígito (izq→der).
    Heurística simple con contornos + padding.
    """
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

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[0])

    # Extrae recortes con padding
    crops = []
    for (x,y,w,h) in boxes[:max_digits]:
        pad = max(2, int(0.15*max(w,h)))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad); y2 = min(gray.shape[0], y + h + pad)
        crops.append(gray[y1:y2, x1:x2])
    return crops

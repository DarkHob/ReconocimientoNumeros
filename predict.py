# ================== PREDICCIÓN ==================
import numpy as np
import cv2
from config import CLASS_DIGITS, IMG_SIZE
from vision import mnist_like_from_gray, segment_digits_from_gray

def _predict_digit_with_tta(model, gray_crop):
    """
    TTA: traslaciones pequeñas y promedio de probabilidades.
    """
    shifts = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
    probs = []
    for dx,dy in shifts:
        arr = mnist_like_from_gray(gray_crop)[...,0]
        M = np.float32([[1,0,dx],[0,1,dy]])
        shifted = cv2.warpAffine((arr*255).astype(np.uint8), M, (IMG_SIZE,IMG_SIZE),
                                 flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)/255.0
        x = np.expand_dims(shifted, axis=(0,-1))
        p = model.predict(x, verbose=0)[0]
        probs.append(p)
    probs = np.mean(probs, axis=0)
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])

def predict_number_from_gray(gray, model):
    """
    Segmenta 1–3 dígitos y concatena. Devuelve (valor:int|None, detalle:str).
    """
    digit_crops = segment_digits_from_gray(gray, max_digits=3)
    if not digit_crops:
        return None, "No se detectaron dígitos."

    preds_str, confs = [], []
    for crop in digit_crops:
        idx, conf = _predict_digit_with_tta(model, crop)
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

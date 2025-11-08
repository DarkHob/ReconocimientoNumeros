# Reconocimiento de Números 

Aplicación de escritorio (Tkinter) que:
- Entrena o carga una CNN (Keras/TensorFlow) para reconocer **dígitos 0–9** con MNIST.
- **Segmenta automáticamente** 1–3 dígitos pegados en una imagen (cámara o lienzo).
- Predice el número **0–100** (concatenando hasta 3 dígitos) con **Test-Time Augmentation (TTA)**.
- **Lee en voz alta** el resultado con `pyttsx3`.
- Ofrece dos pestañas: **Cámara** y **Dibujar**.

---

## 📦 Requisitos

- Python 3.9–3.12  
- Dependencias:
  - `tensorflow`
  - `tensorflow-datasets`
  - `opencv-python`
  - `numpy`
  - `Pillow`
  - `pyttsx3`
  - `tkinter` (viene con Python en Windows/macOS; en Linux se instala como `python3-tk`)

---

## 🚀 Instalación rápida

```bash
python -m venv .venv
```
# Activar entorno
# Windows
```bash
.venv\Scripts\activate
```
# Actualizar pip
```bash
python -m pip install --upgrade pip
```
# Instalar dependencias
```bash
pip install -r requirements.txt
```

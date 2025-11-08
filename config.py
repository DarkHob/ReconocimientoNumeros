# ================== CONFIGURACIÓN GENERAL Y DE "NEURONAS" ==================

# Modelo / dataset
MODEL_PATH   = "mnist_digit_cnn.keras"   # archivo del modelo entrenado
IMG_SIZE     = 28                         # tamaño de entrada (MNIST)
CLASS_DIGITS = [str(i) for i in range(10)]

# Hiperparámetros de entrenamiento
BATCHSIZE = 64
EPOCHS    = 5

# Lienzo de dibujo (UI)
CANVAS_SIZE = 360     # tamaño del cuadro de dibujo en px
BRUSH_SIZE  = 20      # grosor del pincel

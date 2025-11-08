# ================== ENTRENAMIENTO / CARGA DEL MODELO ==================
import os
import tensorflow as tf
import tensorflow_datasets as tfds

from config import MODEL_PATH, BATCHSIZE, EPOCHS, IMG_SIZE
from model_def import build_digit_model

def train_or_load_digit_model():
    if os.path.exists(MODEL_PATH):
        print(f"Cargando modelo desde {MODEL_PATH} ...")
        return tf.keras.models.load_model(MODEL_PATH)

    print("No hay modelo. Entrenando con MNIST (0-9) ...")
    (ds, info) = tfds.load('mnist', as_supervised=True, with_info=True)
    train_ds, test_ds = ds['train'], ds['test']

    def normalize(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, -1)  # (28,28,1)
        return img, label

    train_ds = (train_ds
                .map(normalize)
                .shuffle(10000)
                .batch(BATCHSIZE)
                .prefetch(tf.data.AUTOTUNE))
    test_ds  = (test_ds
                .map(normalize)
                .batch(BATCHSIZE)
                .prefetch(tf.data.AUTOTUNE))

    model = build_digit_model(IMG_SIZE)
    model.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=test_ds)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Accuracy test MNIST: {test_acc:.4f}")
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    return model

# ================== CÁMARAS ==================
import cv2

def list_available_cameras(max_index=10):
    """
    Devuelve índices de cámaras disponibles.
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

class CameraManager:
    def __init__(self):
        self.cap = None

    def open(self, index:int) -> bool:
        self.close()
        self.cap = cv2.VideoCapture(index)
        return self.cap.isOpened()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def is_open(self):
        return self.cap is not None and self.cap.isOpened()

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

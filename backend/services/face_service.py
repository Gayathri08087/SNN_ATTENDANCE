import logging
import json
import time
import cv2
import numpy as np
from config import settings

logger = logging.getLogger("face_service")


class FaceService:
    """
    Uses the SAME trained LBPH model as FACE_MODULE (trainer.yml + labels.json),
    not a fake pixel-flatten embedding. This keeps API-triggered verification
    consistent with what your ESP32-CAM + RFID scanner actually uses.
    """

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(settings.FACE_CASCADE_PATH)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._model_loaded = False
        try:
            self.recognizer.read(settings.FACE_MODEL_PATH)
            with open(settings.FACE_LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = json.load(f)  # {"0": "E0224030", ...}
            self._model_loaded = True
        except Exception as e:
            logger.warning(f"Could not load trained face model yet: {e}. "
                            f"Run FACE_MODULE/Python/train_faces.py first.")
            self.labels = {}

    def _grab_frame(self, timeout=5):
        """Pulls one frame from the ESP32-CAM stream, falling back to the local webcam."""
        cap = cv2.VideoCapture(settings.ESP32_CAM_STREAM_URL)
        if not cap.isOpened():
            logger.warning("ESP32-CAM stream unavailable, falling back to local webcam.")
            cap = cv2.VideoCapture(0)

        start = time.time()
        frame = None
        while time.time() - start < timeout:
            ret, f = cap.read()
            if ret and f is not None:
                frame = f
                break
        cap.release()
        return frame

    def verify_face_for_student(self, expected_student_id: str, timeout: float = 10.0) -> dict:
        """
        Looks for expected_student_id's face using the trained LBPH model.
        Returns {"success": bool, "status": str, "confidence": float|None}
        """
        if not self._model_loaded:
            return {"success": False, "status": "Model not trained yet", "confidence": None}

        start = time.time()
        while time.time() - start < timeout:
            frame = self._grab_frame(timeout=2)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                label, confidence = self.recognizer.predict(face)

                if confidence < settings.FACE_CONFIDENCE_THRESHOLD:
                    predicted_id = self.labels.get(str(label), "Unknown")
                    if predicted_id == expected_student_id:
                        return {
                            "success": True,
                            "status": "Verified",
                            "confidence": round(confidence, 2),
                        }

        return {"success": False, "status": "Unknown Face / No Match", "confidence": None}


face_service = FaceService()

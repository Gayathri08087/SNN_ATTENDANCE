import csv
import json
import os
from datetime import datetime
import cv2

from config import STREAM_URL

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "Models", "trainer.yml")
LABEL_PATH = os.path.join(BASE_DIR, "Models", "labels.json")
CASCADE_PATH = os.path.join(BASE_DIR, "HaarCascade", "haarcascade_frontalface_default.xml")

CSV_DIR = os.path.join(BASE_DIR, "CSV")
os.makedirs(CSV_DIR, exist_ok=True)

CSV_FILE = os.path.join(CSV_DIR, "attendance.csv")

# =========================
# CSV SCHEMA
# =========================
COLUMNS = [
    "Date",
    "Day",
    "Time",
    "Timestamp",
    "Student_ID",
    "Student_Name",
    "Face_Confidence",
    "Face_Status"
]

def write_csv(row):
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in COLUMNS})


# =========================
# MAIN
# =========================
def verify_face(expected_student_id):

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Camera not connected")
        return

    print("===================================")
    print("FACE RECOGNITION STARTED")
    print("===================================")

    # =========================
    # SESSION CONTROL (IMPORTANT)
    # =========================
    session_active = True
    marked_students = set()

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))

        now = datetime.now()

        date = now.strftime("%Y-%m-%d")
        day = now.strftime("%A")
        time_only = now.strftime("%H:%M:%S")
        timestamp = now.isoformat()

        # =========================
        # NO SESSION
        # =========================
        if not session_active:
            cv2.putText(frame, "WAITING FOR RFID...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # =========================
        # NO FACE
        # =========================
        if len(faces) == 0:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # =========================
        # FACE PROCESS
        # =========================
        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < 90:
                student_id = labels[str(label)]

                if student_id == expected_student_id:
                    cap.release()
                    cv2.destroyAllWindows()
                    return {
                       "success": True,
                       "student_id": student_id,
                       "confidence": round(confidence,2),
                       "status":"Verified"
               }
 
            else:
                cap.release() 
                cv2.destroyAllWindows()

                return {
        "success":False,
        "status":"Unknown Face"
    }

if __name__ == "__main__":

    print(
        verify_face(
            input("Expected Student ID : ")
        )
    )
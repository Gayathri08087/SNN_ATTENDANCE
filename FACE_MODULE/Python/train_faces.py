import cv2
import os
import json
import numpy as np

# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

MODEL_DIR = os.path.join(BASE_DIR, "Models")

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "HaarCascade",
    "haarcascade_frontalface_default.xml"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# Load Haar Cascade
# ==========================================

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("ERROR : Haar Cascade not found.")
    exit()

# ==========================================
# Create Recognizer
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

label_map = {}

current_label = 0

print("\n========================================")
print("TRAINING STARTED")
print("========================================\n")

# ==========================================
# Read Dataset
# ==========================================

students = sorted(os.listdir(DATASET_DIR))

for student in students:

    student_path = os.path.join(DATASET_DIR, student)

    if not os.path.isdir(student_path):
        continue

    print("Student :", student)

    label_map[current_label] = student

    for image_name in os.listdir(student_path):

        image_path = os.path.join(student_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(detected) == 0:
            print("  Skipped :", image_name)
            continue

        x, y, w, h = detected[0]

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (200, 200))

        faces.append(face)
        labels.append(current_label)

        print("  Added :", image_name)

    current_label += 1

# ==========================================
# Train Model
# ==========================================

if len(faces) == 0:

    print("\nNo faces found.")
    exit()

recognizer.train(faces, np.array(labels))

# ==========================================
# Save Model
# ==========================================

trainer_path = os.path.join(
    MODEL_DIR,
    "trainer.yml"
)

recognizer.save(trainer_path)

label_path = os.path.join(
    MODEL_DIR,
    "labels.json"
)

with open(label_path, "w") as f:
    json.dump(label_map, f, indent=4)

print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print("Students :", len(label_map))
print("Images   :", len(faces))

print("\nSaved Model :")
print(trainer_path)

print("\nSaved Labels :")
print(label_path)
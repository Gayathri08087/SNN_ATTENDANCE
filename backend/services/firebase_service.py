import logging
import os
import firebase_admin
from firebase_admin import credentials, firestore
from config import settings

logger = logging.getLogger("firebase_service")


class FirebaseService:
    def __init__(self):
        self.db = None
        self._initialize_sdk()

    def _initialize_sdk(self):
        try:
            if not os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Missing service account file at {settings.FIREBASE_CREDENTIALS_PATH}"
                )

            if not firebase_admin._apps:
                cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized.")

            self.db = firestore.client()
            logger.info("Firestore client connected.")
        except Exception as e:
            logger.critical(f"Failed to initialize Firebase: {str(e)}")
            raise e

    # ==========================
    # STUDENTS
    # Schema (matches RFID_MODULE/main.py): collection "students"
    # fields: studentId, name, department, rfidId, face_trained (bool)
    # ==========================

    def get_student(self, student_id: str) -> dict:
        try:
            doc = self.db.collection("students").document(student_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"Error fetching student {student_id}: {str(e)}")
            raise e

    def get_student_by_rfid(self, rfid_uid: str) -> dict:
        try:
            query = self.db.collection("students").where("rfidId", "==", rfid_uid).limit(1).stream()
            for doc in query:
                data = doc.to_dict()
                data["studentId"] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Error querying student by rfidId {rfid_uid}: {str(e)}")
            raise e

    def get_all_students(self) -> list:
        try:
            docs = self.db.collection("students").stream()
            result = []
            for doc in docs:
                data = doc.to_dict()
                data["studentId"] = doc.id
                result.append(data)
            return result
        except Exception as e:
            logger.error(f"Error fetching all students: {str(e)}")
            raise e

    def register_student(self, student_id: str, student_data: dict) -> bool:
        try:
            self.db.collection("students").document(student_id).set(student_data, merge=True)
            logger.info(f"Student {student_id} saved/updated.")
            return True
        except Exception as e:
            logger.error(f"Error saving student {student_id}: {str(e)}")
            raise e

    def delete_student(self, student_id: str) -> bool:
        try:
            self.db.collection("students").document(student_id).delete()
            return True
        except Exception as e:
            logger.error(f"Error deleting student {student_id}: {str(e)}")
            raise e

    # ==========================
    # ATTENDANCE
    # ==========================

    def mark_attendance(self, attendance_id: str, attendance_data: dict) -> bool:
        try:
            self.db.collection("attendance").document(attendance_id).set(attendance_data)
            return True
        except Exception as e:
            logger.error(f"Error writing attendance record: {str(e)}")
            raise e

    def get_attendance_records(self) -> list:
        try:
            docs = self.db.collection("attendance").order_by(
                "date", direction=firestore.Query.DESCENDING
            ).stream()
            records = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                records.append(data)
            return records
        except Exception as e:
            logger.error(f"Error retrieving attendance history: {str(e)}")
            raise e


firebase_service = FirebaseService()

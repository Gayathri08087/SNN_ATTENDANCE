import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.firebase_service import firebase_service

logger = logging.getLogger("students_router")
router = APIRouter()


class StudentRegisterRequest(BaseModel):
    student_id: str = Field(..., description="Unique student ID, e.g. E0224030")
    name: str
    department: str
    rfid_uid: str = Field(..., description="RFID card UID, matches what the reader prints")


class StudentResponse(BaseModel):
    student_id: str
    name: str
    department: str
    rfid_uid: str
    face_trained: bool
    registered_at: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_student(payload: StudentRegisterRequest):
    try:
        if firebase_service.get_student(payload.student_id):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Student ID {payload.student_id} is already registered.",
            )
        if firebase_service.get_student_by_rfid(payload.rfid_uid):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"RFID UID {payload.rfid_uid} is already assigned to another student.",
            )

        student_data = {
            "studentId": payload.student_id,
            "name": payload.name,
            "department": payload.department,
            "rfidId": payload.rfid_uid,
            "face_trained": False,
            "registered_at": datetime.utcnow().isoformat(),
        }

        if not firebase_service.register_student(payload.student_id, student_data):
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Database write failure.")

        return {
            "status": "success",
            "message": f"Registered {payload.student_id}. "
                       f"Next: run FACE_MODULE/Python/capture_faces.py with this same ID, "
                       f"then FACE_MODULE/Python/train_faces.py.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering student: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@router.post("/mark-trained/{student_id}", status_code=status.HTTP_200_OK)
async def mark_face_trained(student_id: str):
    """
    Call this after you've run capture_faces.py + train_faces.py for this
    student, so the roster reflects that their face model is ready.
    """
    try:
        if not firebase_service.get_student(student_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Student {student_id} not found.")

        firebase_service.register_student(student_id, {"face_trained": True})
        return {"status": "success", "message": f"{student_id} marked as face-trained."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating training flag: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@router.get("/", response_model=List[StudentResponse], status_code=status.HTTP_200_OK)
async def list_students():
    try:
        raw = firebase_service.get_all_students()
        return [
            StudentResponse(
                student_id=item.get("studentId", ""),
                name=item.get("name", ""),
                department=item.get("department", ""),
                rfid_uid=item.get("rfidId", ""),
                face_trained=item.get("face_trained", False),
                registered_at=item.get("registered_at", datetime.utcnow().isoformat()),
            )
            for item in raw
        ]
    except Exception as e:
        logger.error(f"Error listing students: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Roster lookup failure.")


@router.delete("/{student_id}", status_code=status.HTTP_200_OK)
async def delete_student(student_id: str):
    try:
        if not firebase_service.get_student(student_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Student {student_id} not found.")
        firebase_service.delete_student(student_id)
        return {"status": "success", "message": f"{student_id} deleted."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting student {student_id}: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))

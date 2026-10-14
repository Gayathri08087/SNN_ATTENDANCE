import logging
from datetime import datetime
from uuid import uuid4
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.firebase_service import firebase_service
from services.face_service import face_service
from services.rfid_service import rfid_service
from services.snn_service import snn_service

logger = logging.getLogger("attendance_router")
router = APIRouter()


class AttendanceLogResponse(BaseModel):
    id: str
    student_id: str
    name: str
    department: str
    date: str
    time: str
    status: str
    confidence: float
    rfid_uid: str


class ManualScanRequest(BaseModel):
    rfid_uid: str = Field(..., description="RFID UID, for testing without physical hardware")


@router.post("/scan", status_code=status.HTTP_200_OK)
async def process_attendance_scan(payload: Optional[ManualScanRequest] = None):
    """
    Full pipeline: RFID lookup -> face verification (real trained LBPH model) ->
    SNN confidence check -> log to Firestore.
    """
    try:
        # Step 1: get UID, either from request body (testing) or live hardware
        scanned_uid = payload.rfid_uid.strip() if (payload and payload.rfid_uid) else rfid_service.read_rfid_uid()
        if not scanned_uid:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No RFID signal received.")

        # Step 2: look up student
        student = firebase_service.get_student_by_rfid(scanned_uid)
        if not student:
            rfid_service.send_acknowledgement(success=False)
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"RFID UID {scanned_uid} is not registered to any student.",
            )

        student_id = student["studentId"]

        if not student.get("face_trained"):
            rfid_service.send_acknowledgement(success=False)
            raise HTTPException(
                status.HTTP_428_PRECONDITION_REQUIRED,
                f"Student {student_id} has no trained face model yet. "
                f"Run capture_faces.py + train_faces.py, then POST /mark-trained/{student_id}.",
            )

        # Step 3: real face verification against the trained LBPH model
        face_result = face_service.verify_face_for_student(student_id, timeout=10.0)
        if not face_result["success"]:
            rfid_service.send_acknowledgement(success=False)
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                f"Face verification failed: {face_result['status']}",
            )

        # Step 4: SNN confidence check (placeholder weights unless you've retrained --
        # see snn_service docstring)
        face_conf_normalized = max(0.0, 1.0 - (face_result["confidence"] or 0) / 100.0)
        snn_valid, snn_confidence = snn_service.verify_transaction(
            face_conf=face_conf_normalized,
            rfid_conf=1.0,
            prev_attendance_rate=0.82,
            time_diff_factor=1.0,
        )

        if not snn_valid:
            rfid_service.send_acknowledgement(success=False)
            raise HTTPException(
                status.HTTP_406_NOT_ACCEPTED,
                f"SNN confidence check below threshold ({snn_confidence:.2f}).",
            )

        # Step 5: log attendance
        timestamp = datetime.utcnow()
        attendance_id = f"ATT_{student_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        record = {
            "student_id": student_id,
            "name": student.get("name"),
            "department": student.get("department"),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "status": "Present",
            "confidence": float(round(snn_confidence, 2)),
            "rfid_uid": scanned_uid,
        }

        if not firebase_service.mark_attendance(attendance_id, record):
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to save attendance.")

        rfid_service.send_acknowledgement(success=True)

        return {
            "status": "success",
            "message": f"Attendance recorded for {student.get('name')}.",
            "data": record,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Attendance scan error: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


@router.get("/", response_model=List[AttendanceLogResponse], status_code=status.HTTP_200_OK)
async def get_attendance_history():
    try:
        raw_logs = firebase_service.get_attendance_records()
        return [
            AttendanceLogResponse(
                id=r.get("id", str(uuid4())),
                student_id=r.get("student_id", "Unknown"),
                name=r.get("name", "Unknown"),
                department=r.get("department", "N/A"),
                date=r.get("date", ""),
                time=r.get("time", ""),
                status=r.get("status", "Present"),
                confidence=r.get("confidence", 0.0),
                rfid_uid=r.get("rfid_uid", ""),
            )
            for r in raw_logs
        ]
    except Exception as e:
        logger.error(f"Error fetching attendance history: {str(e)}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to retrieve history.")

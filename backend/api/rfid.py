# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Query, Body
from services.rfid_service import verify_rfid, determine_rfid_status
from core.session_manager import session
from datetime import datetime

router = APIRouter()

@router.post("/scan")
def scan(rfid_id: str = Query(None), uid: dict = Body(default=None)):
    """
    Perform RFID scan.
    Supports rfid_id as query parameter (used by frontend simulation) and request body (used by devices).
    """
    card = rfid_id
    if not card and uid:
        card = uid.get("uid") or uid.get("rfid_id") or uid.get("rfidId")
        
    if not card:
        return {"status": "ERROR", "message": "No RFID card identifier provided"}

    card = card.strip().upper()
    student = verify_rfid(card)
    
    if student is None:
        # Unregistered RFID scan
        result = session.rfid_scanned(None, card, "Card Not Registered")
        return result

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    status = determine_rfid_status(student["id"], date)
    
    result = session.rfid_scanned(student, card, status)
    return result
import os
import time
from datetime import datetime

import pandas as pd
import serial
from google.cloud.firestore_v1 import FieldFilter

from backend.services.firebase_service import db

# =====================================================
# CONFIGURATION
# =====================================================

COM_PORT = os.getenv("RFID_COM_PORT", "COM3")
BAUD_RATE = int(os.getenv("RFID_BAUD_RATE", "115200"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.getenv(
    "RFID_DATABASE_FILE",
    os.path.join(BASE_DIR, "rfid_database.csv")
)

# =====================================================
# CSV SCHEMA
# =====================================================

STANDARD_COLUMNS = [
    "Date",
    "Day",
    "Time",
    "Timestamp",
    "Student_ID",
    "Student_Name",
    "RFID_UID",
    "RFID_Status",
    "Face_Status",
    "Attendance",
    "Anomaly",
    "Reason"
]

# =====================================================
# STATUS VALUES
# =====================================================

STATUS_VERIFIED = "Verified"
STATUS_ALREADY_MARKED = "Attendance Already Marked"
STATUS_SUSPICIOUS = "Multiple Access Attempt (Suspicious)"
STATUS_NOT_REGISTERED = "Card Not Registered"

# Statuses that count as a scan record for a student on a given day
# when determining the next status. This count is the ONLY input to
# the decision - no timing or debounce windows are used.
COUNTED_STATUSES = (
    STATUS_VERIFIED,
    STATUS_ALREADY_MARKED,
    STATUS_SUSPICIOUS,
)

# =====================================================
# CSV HELPERS
# =====================================================

def create_csv_if_needed(file_path):
    needs_header = (
        not os.path.exists(file_path)
        or os.path.getsize(file_path) == 0
    )

    if needs_header:
        pd.DataFrame(columns=STANDARD_COLUMNS).to_csv(
            file_path,
            index=False
        )


def append_rfid_event(event_data, file_path):
    create_csv_if_needed(file_path)

    row = {
        column: event_data.get(column, "")
        for column in STANDARD_COLUMNS
    }

    pd.DataFrame([row]).to_csv(
        file_path,
        mode="a",
        header=False,
        index=False
    )


def count_records_today(file_path, student_id, date):
    """
    Counts all counted-status records (Verified, Attendance Already
    Marked, Multiple Access Attempt (Suspicious)) for this student on
    today's date. This count is purely history-based - it does not
    depend on time gaps, timestamps, or debounce windows.
    """
    if not os.path.exists(file_path):
        return 0

    try:
        df = pd.read_csv(file_path, dtype=str)
    except Exception:
        return 0

    if df.empty:
        return 0

    mask = (
        (df["Student_ID"].astype(str) == str(student_id))
        & (df["Date"].astype(str) == str(date))
        & (df["RFID_Status"].isin(COUNTED_STATUSES))
    )

    return int(mask.sum())


def determine_rfid_status(file_path, student_id, date):
    """
    Decides the RFID_Status for a registered card purely by counting
    prior records for this student on this date:

    - 0 prior records -> Verified
    - 1 prior record  -> Attendance Already Marked
    - 2+ prior records -> Multiple Access Attempt (Suspicious)
    """
    prior_count = count_records_today(file_path, student_id, date)

    if prior_count == 0:
        return STATUS_VERIFIED
    elif prior_count == 1:
        return STATUS_ALREADY_MARKED
    else:
        return STATUS_SUSPICIOUS


# =====================================================
# FIREBASE HELPERS
# =====================================================

def lookup_student_by_uid(uid):
    docs = db.collection("students").where(
        filter=FieldFilter("rfidId", "==", uid)
    ).stream()

    for doc in docs:
        return doc

    return None


# =====================================================
# FACE MODULE HANDOFF
# =====================================================

def launch_face_recognition(student_id, student_name):
    print("\n====================================")
    print("RFID VERIFIED")
    print("Launching Face Recognition...")
    print(f"Student : {student_name}")
    print(f"ID      : {student_id}")
    print("====================================\n")


# =====================================================
# EVENT HANDLERS
# =====================================================

def build_event(date, day, time_only, timestamp, uid,
                 student_id, student_name, status):
    return {
        "Date": date,
        "Day": day,
        "Time": time_only,
        "Timestamp": timestamp,
        "RFID_UID": uid,
        "Student_ID": student_id,
        "Student_Name": student_name,
        "RFID_Status": status,
    }


def handle_not_registered(uid, date, day, time_only, timestamp, file_path):
    event_data = build_event(
        date, day, time_only, timestamp,
        uid, "Unknown", "Unknown", STATUS_NOT_REGISTERED
    )
    append_rfid_event(event_data, file_path)

    print("\n==============================")
    print("CARD NOT REGISTERED (ANOMALY)")
    print("==============================")
    print(f"UID : {uid}")
    print("Status : Card Not Registered")
    print("==============================\n")

    return STATUS_NOT_REGISTERED


def handle_authorized(student, uid, date, day, time_only, timestamp, file_path):
    data = student.to_dict()

    # Use the Student ID field instead of Firestore document ID
    student_id = data.get("studentId", "Unknown")

    # Use the correct field name from Firestore
    student_name = data.get("name", "Unknown")

    status = determine_rfid_status(file_path, student_id, date)

    event_data = build_event(
        date,
        day,
        time_only,
        timestamp,
        uid,
        student_id,
        student_name,
        status,
    )

    append_rfid_event(event_data, file_path)

    print("\n==============================")

    if status == STATUS_VERIFIED:
        print("AUTHORIZED USER")
    elif status == STATUS_ALREADY_MARKED:
        print("ATTENDANCE ALREADY MARKED")
    else:
        print("MULTIPLE ACCESS ATTEMPT (SUSPICIOUS)")

    print("==============================")
    print(f"UID       : {uid}")
    print(f"StudentID : {student_id}")
    print(f"Name      : {student_name}")
    print(f"Time      : {time_only}")
    print(f"Status    : {status}")
    print("==============================\n")

    if status == STATUS_VERIFIED:
        launch_face_recognition(student_id, student_name)

    return status


# =====================================================
# MAIN
# =====================================================

def main():
    create_csv_if_needed(DATABASE_FILE)

    print("===================================")
    print(" RFID ATTENDANCE LOGGER STARTED")
    print(" Connected to Firebase")
    print(" Waiting for RFID Cards...")
    print("===================================")

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
    except Exception as e:
        print("Could not connect to ESP32")
        print(e)
        return

    while True:
        try:
            if not ser.in_waiting:
                continue

            line = ser.readline().decode(errors="ignore").strip()

            if not line.startswith("UID:"):
                continue

            uid = line.replace("UID:", "").strip().upper()

            now = datetime.now()

            date = now.strftime("%Y-%m-%d")
            day = now.strftime("%A")
            time_only = now.strftime("%H:%M:%S")
            timestamp = now.isoformat()

            # ====================================
            # FIREBASE LOOKUP
            # ====================================
            student = lookup_student_by_uid(uid)

            if not student:
                handle_not_registered(
                    uid, date, day, time_only, timestamp, DATABASE_FILE
                )
                # Send Access Denied code '0' back to hardware module
                ser.write(b'0')
                continue

            # Capture authorization status string
            status = handle_authorized(
                student, uid, date, day, time_only, timestamp, DATABASE_FILE
            )

            # Evaluate status output and provide active hardware visual/auditory cues
            if status == STATUS_VERIFIED:
                ser.write(b'1')  # Access Granted: Flash Green + Beep Twice
            else:
                ser.write(b'0')  # Logged/Suspicious: Flash Red + Long Solid Error Tone

        except KeyboardInterrupt:
            print("\nProgram Stopped")
            ser.close()
            break

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
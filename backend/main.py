# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api import rfid, face, session, students, attendance, snn
from services.rfid_service import rfid_listener

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start RFID Background Listener thread
    try:
        rfid_listener.start()
    except Exception as e:
        print(f"Failed to start RFID listener thread: {e}")
    yield
    # Shutdown: Stop RFID Background Listener thread
    try:
        rfid_listener.stop()
    except Exception as e:
        print(f"Failed to stop RFID listener thread: {e}")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes with prefixes to match frontend API queries
app.include_router(session.router, prefix="/session", tags=["Session"])
app.include_router(rfid.router, prefix="/rfid", tags=["RFID"])
app.include_router(face.router, prefix="/face", tags=["Face"])
app.include_router(students.router)  # Prefix is defined as "/students" inside students.py
app.include_router(attendance.router, prefix="/attendance", tags=["Attendance"])
app.include_router(snn.router, prefix="/snn", tags=["SNN"])

@app.get("/")
def home():
    return {"status": "SNN ATTENDANCE SYSTEM RUNNING"}
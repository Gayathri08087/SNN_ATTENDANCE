import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


def _find_firebase_key() -> str:
    """Tolerates the 'firebse' typo folder as well as the correct 'firebase' spelling."""
    candidates = [
        BASE_DIR / "firebase" / "firebase_key.json",
        BASE_DIR / "firebse" / "firebase_key.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    # Default to the correctly-spelled path even if missing yet, so the error
    # message on startup tells you exactly what to create/rename.
    return str(candidates[0])


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    SECRET_KEY: str = "SUPER_SECRET_PRODUCTION_READY_KEY_CHANGE_THIS"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    FIREBASE_CREDENTIALS_PATH: str = _find_firebase_key()

    SERIAL_PORT: str = os.getenv("RFID_COM_PORT", "COM3")
    BAUD_RATE: int = int(os.getenv("RFID_BAUD_RATE", "115200"))
    ESP32_CAM_IP: str = "10.112.228.11"
    ESP32_CAM_STREAM_URL: str = f"http://{ESP32_CAM_IP}:81/stream"

    FACE_DATASET_DIR: str = str(BASE_DIR.parent / "FACE_MODULE" / "Dataset")
    FACE_MODEL_PATH: str = str(BASE_DIR.parent / "FACE_MODULE" / "Models" / "trainer.yml")
    FACE_LABELS_PATH: str = str(BASE_DIR.parent / "FACE_MODULE" / "Models" / "labels.json")
    FACE_CASCADE_PATH: str = str(BASE_DIR.parent / "FACE_MODULE" / "HaarCascade" / "haarcascade_frontalface_default.xml")
    SNN_MODEL_DIR: str = str(BASE_DIR.parent / "SNN_MODULE" / "Models")

    FACE_CONFIDENCE_THRESHOLD: float = 90.0   # LBPH: LOWER value = better match
    SNN_CONFIDENCE_THRESHOLD: float = 0.85

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

if not os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
    print(f"!! WARNING: Firebase key not found at {settings.FIREBASE_CREDENTIALS_PATH}")
    print("!! Rename your key folder to 'backend/firebase/firebase_key.json'.")

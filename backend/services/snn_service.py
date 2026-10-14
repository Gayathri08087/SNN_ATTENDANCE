import os
import sys
import torch
import pandas as pd
from datetime import datetime
from config import Config

# Add SNN_MODULE to Python sys.path so we can import modules from it
snn_module_path = os.path.join(Config.BASE_DIR, "SNN_MODULE")
if snn_module_path not in sys.path:
    sys.path.append(snn_module_path)

# Lazy import helpers for SNN training/preprocessing to prevent circular dependencies
def get_snn_classifier():
    """Return the SpikingClassifier model class definition."""
    try:
        from snn_model import SpikingClassifier
        return SpikingClassifier
    except ImportError as e:
        print(f"Failed to import SpikingClassifier: {e}")
        return None

def calculate_anomaly_fallback(rfid_status: str, face_status: str) -> float:
    """Simple rule-based anomaly score fallback in case PyTorch model is missing."""
    rf_status_lower = str(rfid_status).lower()
    fc_status_lower = str(face_status).lower()
    
    if "duplicate" in rf_status_lower or "duplicate" in fc_status_lower or "suspicious" in rf_status_lower:
        return 0.9
    if "verified" in rf_status_lower and "verified" in fc_status_lower:
        return 0.1
    if "not registered" in rf_status_lower or "unknown" in fc_status_lower:
        return 0.8
    return 0.5

def encode_status_values(rfid_status: str, face_status: str):
    """Encode RFID and Face status terms to floats as expected by SNN preprocessing."""
    # Matches SNN_MODULE/preprocess.py mapping
    rfid_map = {
        "verified": 1.0,
        "card not registered": 0.0,
        "duplicate scan": 0.0,
        "multiple access attempt (suspicious)": 0.0
    }
    face_map = {
        "verified": 1.0,
        "duplicate": 0.0,
        "unknown": 0.0,
        "no face detected": 0.0
    }
    
    rf_val = rfid_map.get(str(rfid_status).lower().strip(), -1.0)
    fc_val = face_map.get(str(face_status).lower().strip(), -1.0)
    return rf_val, fc_val

def time_to_minutes(time_str: str) -> float:
    """Convert HH:MM:SS string to minutes since midnight."""
    try:
        if ":" in str(time_str):
            parts = str(time_str).split(":")
            hh = int(parts[0])
            mm = int(parts[1])
            return float(hh * 60 + mm)
        return 0.0
    except ValueError:
        return 0.0

def predict_anomaly_score(rfid_status: str, face_status: str, face_confidence: float, time_rfid: str, time_face: str) -> float:
    """
    Run PyTorch SNN inference to determine the anomaly probability of the checkpoint scan.
    Returns a score between 0.0 (Normal) and 1.0 (Highly Suspicious/Anomaly).
    """
    if not os.path.exists(Config.SNN_MODEL_PATH):
        print("Trained SNN model not found. Using rule-based fallback calculation.")
        return calculate_anomaly_fallback(rfid_status, face_status)
    
    ClassifierClass = get_snn_classifier()
    if ClassifierClass is None:
        return calculate_anomaly_fallback(rfid_status, face_status)

    try:
        # Load the model
        # Input size matches the 10 features in training_dataset.csv
        model = ClassifierClass(input_size=10, hidden_size=8, output_size=1)
        model.load_state_dict(torch.load(Config.SNN_MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        
        # Prepare inputs
        rf_val, fc_val = encode_status_values(rfid_status, face_status)
        t_rfid_num = time_to_minutes(time_rfid)
        t_face_num = time_to_minutes(time_face)
        
        # Feature columns list ordered:
        # Timestamp_face (0), Student_ID_face (1), Face_Confidence (2), Face_Status (3), Timestamp_rfid (4), RFID_UID (5), Student_ID_rfid (6), RFID_Status (7), time_face_num (8), time_rfid_num (9)
        # Note: Categorical strings are parsed as float NaNs and filled with 0.0 in training, so we supply 0.0 for those.
        features = [
            0.0,            # Timestamp_face
            0.0,            # Student_ID_face
            float(face_confidence),
            fc_val,         # Face_Status
            0.0,            # Timestamp_rfid
            0.0,            # RFID_UID
            0.0,            # Student_ID_rfid
            rf_val,         # RFID_Status
            t_face_num,     # time_face_num
            t_rfid_num      # time_rfid_num
        ]
        
        feature_tensor = torch.tensor([features], dtype=torch.float32)
        
        with torch.no_grad():
            # Match train.py dimension mapping: shape [batch_size, features, time_steps]
            spike_tensor = feature_tensor.unsqueeze(2).repeat(1, 1, 3)
            outputs, _ = model(spike_tensor)
            probability = torch.sigmoid(outputs.mean(dim=1).sum(dim=1, keepdim=True)).item()
            
        # probability is the classification of "Legitimate Presentation" (Absent=0, Present=1)
        # Anomaly score is 1.0 - probability (higher means more suspicious)
        anomaly_score = 1.0 - probability
        return round(anomaly_score, 4)
    except Exception as e:
        print(f"Error executing SNN inference model: {e}. Falling back to rules.")
        return calculate_anomaly_fallback(rfid_status, face_status)

def run_snn_pipeline():
    """Execute SNN preprocessing and SNN retraining pipelines consecutively."""
    try:
        # pyrefly: ignore [missing-import]
        from preprocess import main as preprocess_main
        from train_snn import main as train_main
        
        preprocess_main()
        train_main()
        return True, "SNN Preprocessing & Model Training completed successfully."
    except Exception as e:
        return False, f"Pipeline execution failed: {str(e)}"
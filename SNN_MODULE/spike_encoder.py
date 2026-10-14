import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def find_project_root() -> Path:
    """Return the project root based on the current file location."""
    return Path(__file__).resolve().parent.parent


def load_training_data(project_root: Path) -> pd.DataFrame:
    """Load the processed training dataset created by preprocess.py."""
    dataset_path = project_root / "SNN_MODULE" / "CSV" / "training_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("Training dataset is empty.")
    return df


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Convert the feature columns into a float32 numpy array."""
    feature_columns = [
        col for col in df.columns
        if col not in {"Attendance_Status_face", "Attendance_Status_rfid"}
    ]

    feature_frame = df[feature_columns].copy()
    for col in feature_frame.columns:
        if not pd.api.types.is_numeric_dtype(feature_frame[col]):
            feature_frame[col] = feature_frame[col].astype(str).str.strip()
            feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")

    feature_frame = feature_frame.fillna(0.0)
    features = feature_frame.to_numpy(dtype=np.float32)
    return features


def prepare_targets(df: pd.DataFrame) -> np.ndarray:
    """Create binary targets from attendance status columns."""
    target_values = []
    for col in ["Attendance_Status_face", "Attendance_Status_rfid"]:
        if col in df.columns:
            target_values.append(df[col].astype(str).str.strip().map({"Present": 1, "Absent": 0, "1": 1, "0": 0}).fillna(0))

    if not target_values:
        raise ValueError("No attendance target columns were found in the dataset.")

    target_array = pd.concat(target_values, axis=1).mean(axis=1).to_numpy(dtype=np.float32)
    target_array = np.clip(target_array, 0.0, 1.0)
    return target_array


def encode_to_spikes(features: np.ndarray, time_steps: int = 10, threshold: float = 0.5) -> torch.Tensor:
    """Encode continuous feature values into spike trains using a simple threshold-based encoder."""
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array.")

    if time_steps <= 0:
        raise ValueError("time_steps must be greater than zero.")

    normalized = np.clip(features, 0.0, 1.0)
    normalized_tensor = torch.from_numpy(normalized).float()
    spike_tensor = torch.zeros((features.shape[0], features.shape[1], time_steps), dtype=torch.float32)

    for step in range(time_steps):
        probability = normalized_tensor * (step + 1) / time_steps
        spike_tensor[:, :, step] = torch.rand_like(probability) < probability

    return spike_tensor


def save_encoded_data(spikes: torch.Tensor, targets: np.ndarray, project_root: Path) -> Path:
    """Save the encoded spike tensor and target values for later training."""
    output_dir = project_root / "SNN_MODULE" / "Models"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"spikes": spikes, "targets": torch.from_numpy(targets)}, output_dir / "encoded_data.pt")
    return output_dir / "encoded_data.pt"


def main() -> None:
    """Load dataset, encode features into spike trains, and save encoded tensors."""
    try:
        project_root = find_project_root()
        df = load_training_data(project_root)
        features = prepare_features(df)
        targets = prepare_targets(df)
        spikes = encode_to_spikes(features)
        output_path = save_encoded_data(spikes, targets, project_root)
        print(f"Spike encoding complete. Encoded file saved at: {output_path}")
    except Exception as exc:
        print(f"Spike encoding failed: {exc}")
        raise


if __name__ == "__main__":
    main()

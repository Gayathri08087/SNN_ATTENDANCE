import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from snn_model import SpikingClassifier


def find_project_root() -> Path:
    """Return the project root based on the current file location."""
    return Path(__file__).resolve().parent.parent


def load_dataset(project_root: Path) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Load the preprocessed CSV file and convert it into tensors for training."""
    dataset_path = project_root / "SNN_MODULE" / "CSV" / "training_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("Training dataset is empty.")

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
    features = torch.tensor(feature_frame.to_numpy(dtype=float), dtype=torch.float32)

    target_values = []
    for col in ["Attendance_Status_face", "Attendance_Status_rfid"]:
        if col in df.columns:
            target_values.append(df[col].astype(str).str.strip().map({"Present": 1, "Absent": 0, "1": 1, "0": 0}).fillna(0))

    if not target_values:
        raise ValueError("No target columns were found in the training dataset.")

    labels = pd.concat(target_values, axis=1).mean(axis=1).to_numpy(dtype=float)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return df, features, labels


def train_model(features: torch.Tensor, labels: torch.Tensor, epochs: int = 5) -> nn.Module:
    """Train a compact spiking classifier for a few epochs on the provided dataset."""
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Features and labels must contain the same number of samples.")

    model = SpikingClassifier(input_size=features.shape[1], hidden_size=8, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        spike_tensor = features.unsqueeze(2).repeat(1, 1, 3)
        outputs, _ = model(spike_tensor)
        output_probs = torch.sigmoid(outputs.mean(dim=1).sum(dim=1, keepdim=True))
        loss = criterion(output_probs, labels)
        loss.backward()
        optimizer.step()

    return model


def save_model(model: nn.Module, project_root: Path) -> Path:
    """Save the trained model to the SNN_MODULE/Models directory."""
    model_dir = project_root / "SNN_MODULE" / "Models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "trained_snn.pt"
    torch.save(model.state_dict(), model_path)
    return model_path


def main() -> None:
    """Load data, train the SNN, and save the model."""
    try:
        project_root = find_project_root()
        _, features, labels = load_dataset(project_root)
        model = train_model(features, labels, epochs=3)
        model_path = save_model(model, project_root)
        print(f"Training complete. Model saved at: {model_path}")
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise


if __name__ == "__main__":
    main()

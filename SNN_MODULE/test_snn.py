from pathlib import Path

import pandas as pd
import torch

from snn_model import SpikingClassifier


def find_project_root() -> Path:
    """Return the project root based on the current file location."""
    return Path(__file__).resolve().parent.parent


def get_feature_columns(project_root: Path) -> list[str]:
    """Read the processed training dataset and return its feature-column names."""
    dataset_path = project_root / "SNN_MODULE" / "CSV" / "training_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    feature_columns = [
        col for col in df.columns
        if col not in {"Attendance_Status_face", "Attendance_Status_rfid"}
    ]
    if not feature_columns:
        raise ValueError("No feature columns were found in the training dataset.")
    return feature_columns


def build_sample_input(project_root: Path, feature_columns: list[str]) -> torch.Tensor:
    """Create a sample feature vector that matches the training dataset schema."""
    defaults = {
        "Face_Confidence_face": 84.88,
        "Authorized_face": 1.0,
        "Scan_Result_face": 1.0,
        "Day": 1.0,
        "Authorized_rfid": 1.0,
        "Scan_Result_rfid": 1.0,
        "Face_Confidence_rfid": 84.88,
        "date_face_num": 2026.0,
        "date_rfid_num": 2026.0,
        "time_face_num": 778.0,
        "time_rfid_num": 1365.0,
    }

    values = [defaults.get(col, 0.0) for col in feature_columns]
    return torch.tensor([values], dtype=torch.float32)


def load_model(project_root: Path) -> tuple[SpikingClassifier, torch.Tensor]:
    """Load the trained SNN model from the Models directory."""
    model_path = project_root / "SNN_MODULE" / "Models" / "trained_snn.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    feature_columns = get_feature_columns(project_root)
    model = SpikingClassifier(input_size=len(feature_columns), hidden_size=8, output_size=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    sample_input = build_sample_input(project_root, feature_columns)
    return model, sample_input


def predict_attendance(model: SpikingClassifier, features: torch.Tensor) -> float:
    """Run a forward pass and convert the output to a probability-like prediction."""
    with torch.no_grad():
        spike_tensor = features.unsqueeze(2).repeat(1, 1, 3)
        outputs, _ = model(spike_tensor)
        probability = torch.sigmoid(outputs.mean(dim=1).sum(dim=1, keepdim=True)).item()
    return probability


def main() -> None:
    """Load the trained model and predict attendance for a sample input."""
    try:
        project_root = find_project_root()
        model, sample_input = load_model(project_root)
        probability = predict_attendance(model, sample_input)
        prediction = "Present" if probability >= 0.5 else "Absent"
        print(f"Prediction probability: {probability:.4f}")
        print(f"Predicted attendance: {prediction}")
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        raise


if __name__ == "__main__":
    main()

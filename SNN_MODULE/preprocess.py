import pandas as pd
from pathlib import Path


def find_project_root() -> Path:
    """Return the project root based on the current file location."""
    return Path(__file__).resolve().parent.parent


def load_rfid_data(project_root: Path) -> pd.DataFrame:
    """Load RFID data from the protected RFID module CSV file."""
    rfid_path = project_root / "RFID_MODULE" / "Python" / "rfid_database.csv"
    if not rfid_path.exists():
        raise FileNotFoundError(f"RFID data file not found: {rfid_path}")

    df = pd.read_csv(rfid_path)
    if df.empty:
        raise ValueError("RFID dataset is empty.")

    df.columns = [col.strip() for col in df.columns]
    return df


def load_face_data(project_root: Path) -> pd.DataFrame:
    """Load face attendance data from the protected FACE_MODULE CSV file."""
    face_path = project_root / "FACE_MODULE" / "CSV" / "attendance.csv"
    if not face_path.exists():
        raise FileNotFoundError(f"Face attendance file not found: {face_path}")

    df = pd.read_csv(face_path)
    if df.empty:
        raise ValueError("Face attendance dataset is empty.")

    df.columns = [col.strip() for col in df.columns]
    return df


def merge_datasets(rfid_df: pd.DataFrame, face_df: pd.DataFrame) -> pd.DataFrame:
    """Merge RFID and face datasets using Session_ID as the join key."""
    rfid_df = rfid_df.copy()
    face_df = face_df.copy()

    required_rfid_columns = {"Session_ID", "Student_ID", "Student_Name", "RFID_Status", "Reason", "Authentication_Source"}
    required_face_columns = {"Session_ID", "Student_ID", "Student_Name", "Face_Confidence", "Face_Status", "Reason", "Authentication_Source"}

    if not required_rfid_columns.issubset(rfid_df.columns):
        raise ValueError("RFID dataset does not contain the required standardized columns.")

    if not required_face_columns.issubset(face_df.columns):
        raise ValueError("Face dataset does not contain the required standardized columns.")

    rfid_df["Session_ID"] = rfid_df["Session_ID"].astype(str).str.strip()
    face_df["Session_ID"] = face_df["Session_ID"].astype(str).str.strip()

    merged = pd.merge(
        face_df,
        rfid_df,
        on="Session_ID",
        how="inner",
        suffixes=("_face", "_rfid"),
    )

    if merged.empty:
        raise ValueError("No matching records were found between RFID and face logs using Session_ID.")

    return merged


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical values to numeric values suitable for an SNN model."""
    encoded = df.copy()

    if "Date_face" in encoded.columns:
        encoded["Date_face"] = encoded["Date_face"].astype(str).str.strip()
    if "Date_rfid" in encoded.columns:
        encoded["Date_rfid"] = encoded["Date_rfid"].astype(str).str.strip()

    if "Time_face" in encoded.columns:
        encoded["Time_face"] = encoded["Time_face"].astype(str).str.strip()
    if "Time_rfid" in encoded.columns:
        encoded["Time_rfid"] = encoded["Time_rfid"].astype(str).str.strip()

    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    for col in ["Day_face", "Day_rfid", "Day"]:
        if col in encoded.columns:
            encoded[col] = encoded[col].astype(str).str.strip().map(day_map).fillna(-1)

    if "RFID_Status" in encoded.columns:
        encoded["RFID_Status"] = encoded["RFID_Status"].astype(str).str.strip().map({
            "Verified": 1,
            "Card Not Registered": 0,
            "Duplicate Scan": 0,
        }).fillna(-1)

    if "Face_Status" in encoded.columns:
        encoded["Face_Status"] = encoded["Face_Status"].astype(str).str.strip().map({
            "Verified": 1,
            "Duplicate": 0,
            "Unknown": 0,
            "No Face Detected": 0,
        }).fillna(-1)

    if "Face_Confidence" in encoded.columns:
        encoded["Face_Confidence"] = pd.to_numeric(encoded["Face_Confidence"], errors="coerce").fillna(0.0)

    def time_to_minutes(value: str) -> int:
        try:
            time_part = str(value).strip()
            if ":" in time_part:
                hh, mm, *rest = time_part.split(":")
                return int(hh) * 60 + int(mm)
            return 0
        except ValueError:
            return 0

    if "Time_face" in encoded.columns:
        encoded["time_face_num"] = encoded["Time_face"].apply(time_to_minutes)
    if "Time_rfid" in encoded.columns:
        encoded["time_rfid_num"] = encoded["Time_rfid"].apply(time_to_minutes)

    drop_columns = ["Session_ID", "Date_face", "Date_rfid", "Time_face", "Time_rfid", "Day_face", "Day_rfid", "Student_Name_face", "Student_Name_rfid", "Reason_face", "Reason_rfid", "Authentication_Source_face", "Authentication_Source_rfid"]
    encoded = encoded.drop(columns=[c for c in drop_columns if c in encoded.columns], errors="ignore")

    return encoded


def save_dataset(df: pd.DataFrame, project_root: Path) -> Path:
    """Save the processed dataset to the SNN_MODULE/CSV directory."""
    output_dir = project_root / "SNN_MODULE" / "CSV"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_dataset.csv"
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    """Run the preprocessing pipeline and create the merged training dataset."""
    try:
        project_root = find_project_root()
        rfid_df = load_rfid_data(project_root)
        face_df = load_face_data(project_root)
        merged_df = merge_datasets(rfid_df, face_df)
        processed_df = encode_features(merged_df)
        output_path = save_dataset(processed_df, project_root)
        print(f"Preprocessing complete. Dataset saved at: {output_path}")
    except Exception as exc:
        print(f"Preprocessing failed: {exc}")
        raise


if __name__ == "__main__":
    main()

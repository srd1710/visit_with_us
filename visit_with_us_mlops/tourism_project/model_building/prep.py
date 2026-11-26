from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve project directories independently of the current working directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "splits"

TARGET_COLUMN = "ProdTaken"


def main():
    print("Starting data preparation...")

    # Validate dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    # Remove identifier/index columns if present
    df = df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save splits
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_frame().to_csv(OUTPUT_DIR /

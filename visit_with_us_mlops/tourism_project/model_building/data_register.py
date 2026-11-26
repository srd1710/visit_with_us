"""Validate and summarize the tourism dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"
TARGET = "ProdTaken"
REQUIRED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch",
}

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if TARGET not in df:
        raise ValueError(f"Target column {TARGET!r} is missing")
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("ProdTaken must contain only 0/1 values")
    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("Target distribution:")
    print(df[TARGET].value_counts(dropna=False).sort_index().to_string())
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print("Dataset registration/validation completed.")

if __name__ == "__main__":
    main()

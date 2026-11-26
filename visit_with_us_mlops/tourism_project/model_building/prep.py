"""Clean data and create reproducible train/test CSV splits."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DATA_PATH = ROOT / "data" / "tourism.csv"
SPLIT_DIR = REPO_ROOT / "artifacts" / "splits"
TARGET = "ProdTaken"
DROP_COLUMNS = ["CustomerID", "Unnamed: 0"]  # identifiers/index are not predictive features

def main():
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df:
        raise ValueError(f"Target column {TARGET!r} not found")
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna(subset=[TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("ProdTaken must be binary (0/1)")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(SPLIT_DIR / "Xtrain.csv", index=False)
    X_test.to_csv(SPLIT_DIR / "Xtest.csv", index=False)
    y_train.to_frame(TARGET).to_csv(SPLIT_DIR / "ytrain.csv", index=False)
    y_test.to_frame(TARGET).to_csv(SPLIT_DIR / "ytest.csv", index=False)
    print(f"Saved train/test splits to {SPLIT_DIR}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Features: {len(X.columns)}")

if __name__ == "__main__":
    main()

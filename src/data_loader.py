from pathlib import Path
import pandas as pd

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_reviews() -> pd.DataFrame:
    path = get_project_root() / "data" / "raw" / "Reviews.csv"
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded dataset with {len(df):,} rows.")
    return df

if __name__ == "__main__":
    print(load_reviews().head())

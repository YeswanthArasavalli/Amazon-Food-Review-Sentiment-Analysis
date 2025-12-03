from pathlib import Path
import pandas as pd
from data_loader import load_reviews, get_project_root

def prepare_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Score"] != 3].copy()
    df["label"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
    df = df[["Text", "label"]].rename(columns={"Text": "text"})
    return df

def save_processed(df: pd.DataFrame):
    out = get_project_root() / "data" / "processed" / "reviews_processed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved to {out}")

if __name__ == "__main__":
    df = load_reviews()
    processed = prepare_sentiment(df)
    save_processed(processed)

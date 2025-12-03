import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_loader import get_project_root

def load_processed():
    path = get_project_root() / "data" / "processed" / "reviews_processed.csv"
    return pd.read_csv(path)

if __name__ == "__main__":
    df = load_processed()
    X, y = df["text"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    
    print("Baseline Results:\n")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
    print("\n", classification_report(y_test, preds))

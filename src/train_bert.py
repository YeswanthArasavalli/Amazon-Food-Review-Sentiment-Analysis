import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from data_loader import get_project_root


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "distilbert-base-uncased"
MAX_LEN = 128
BATCH = 16
EPOCHS = 2

def load_processed():
    return pd.read_csv(get_project_root() / "data" / "processed" / "reviews_processed.csv")

class ReviewDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        self.X, self.y, self.tokenizer = X, y, tokenizer

    def __getitem__(self, idx):
        enc = self.tokenizer(self.X[idx], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.y[idx])

    def __len__(self):
        return len(self.X)

if __name__ == "__main__":
    df = load_processed()

    X_train, X_test, y_train, y_test = train_test_split(df.text.values, df.label.values, stratify=df.label, test_size=0.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL)

    train_loader = DataLoader(ReviewDataset(X_train, y_train, tokenizer), batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(ReviewDataset(X_test, y_test, tokenizer), batch_size=BATCH)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(EPOCHS):
        model.train()
        for batch, labels in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)

            loss = model(**batch, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    preds, labels = [], []
    for batch, y in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch).logits.argmax(axis=1).cpu()
        preds.extend(out)
        labels.extend(y)

    print("\n BERT Results:")
    print("Accuracy:", accuracy_score(labels, preds))
    print("F1:", f1_score(labels, preds))

    model.save_pretrained("distilbert_amazon_food")
    tokenizer.save_pretrained("distilbert_amazon_food")

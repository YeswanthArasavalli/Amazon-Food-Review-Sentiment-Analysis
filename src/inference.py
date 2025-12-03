import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = "models/distilbert_amazon_food"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output = model(**enc.to(device)).logits
    probs = torch.softmax(output, dim=1)[0]
    label = "Positive" if torch.argmax(probs).item() == 1 else "Negative"
    return f"{label} (Confidence: {probs.max().item():.2f})"

# Test
print(predict("The food was terrible and cold."))
print(predict("Loved it! Will definitely order again!"))

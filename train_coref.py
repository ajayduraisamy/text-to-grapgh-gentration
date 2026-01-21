import json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

MAX_LEN = 64
EPOCHS = 5
BATCH_SIZE = 8

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------------- DATASET ----------------
class CorefDataset(Dataset):
    def __init__(self, path):
        self.data = json.load(open(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        encoding["labels"] = torch.tensor(1)  
        return {k:v.squeeze() for k,v in encoding.items()}

# ---------------- LOAD ----------------
dataset = CorefDataset("coref_dataset.json")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

optimizer = AdamW(model.parameters(), lr=2e-5)

# ---------------- TRAIN ----------------
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

model.save_pretrained("coref_model")
tokenizer.save_pretrained("coref_model")
print("Coreference model saved")

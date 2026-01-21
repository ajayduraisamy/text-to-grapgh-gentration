import json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

REL_MAP = {
    "works_at":0,
    "located_in":1,
    "lives_in":2,
    "joined":3,
    "studies_at":4
}

MAX_LEN = 32
EPOCHS = 5
BATCH_SIZE = 8

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------------- DATASET ----------------
class RelationDataset(Dataset):
    def __init__(self, path):
        self.data = json.load(open(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = tokenizer(
            item["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        label = REL_MAP[item["triples"][0]["relation"]]
        encoding["labels"] = torch.tensor(label)
        return {k:v.squeeze() for k,v in encoding.items()}

# ---------------- LOAD ----------------
dataset = RelationDataset("relation_dataset.json")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(REL_MAP)
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

model.save_pretrained("relation_model")
tokenizer.save_pretrained("relation_model")
print("Relation model saved")

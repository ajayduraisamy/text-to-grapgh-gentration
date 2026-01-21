import json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.optim import AdamW

# ---------------- CONFIG ----------------
LABEL_MAP = {
    "O": 0,
    "B-PERSON": 1,
    "B-ORG": 2,
    "B-LOCATION": 3
}

MAX_LEN = 32
EPOCHS = 5
BATCH_SIZE = 8

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ---------------- DATASET ----------------
class NERDataset(Dataset):
    def __init__(self, path):
        self.data = json.load(open(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["sentence"].split()

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        labels = [LABEL_MAP["O"]] * MAX_LEN

        for ent in item["entities"]:
            if ent["text"] in words:
                pos = words.index(ent["text"])
                tag = "B-" + ent["label"]   # PERSON → B-PERSON
                labels[pos + 1] = LABEL_MAP[tag]

        encoding["labels"] = torch.tensor(labels)
        return {k: v.squeeze() for k, v in encoding.items()}

# ---------------- LOAD ----------------
dataset = NERDataset("entity_dataset.json")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(LABEL_MAP)
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

model.save_pretrained("ner_model")
tokenizer.save_pretrained("ner_model")
print("NER model saved")

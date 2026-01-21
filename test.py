import torch
import networkx as nx
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    BertTokenizer,
    BertForSequenceClassification
)



ner_tokenizer = BertTokenizerFast.from_pretrained("ner_model")
ner_model = BertForTokenClassification.from_pretrained("ner_model")
ner_model.eval()

coref_tokenizer = BertTokenizer.from_pretrained("coref_model")
coref_model = BertForSequenceClassification.from_pretrained("coref_model")
coref_model.eval()

rel_tokenizer = BertTokenizer.from_pretrained("relation_model")
rel_model = BertForSequenceClassification.from_pretrained("relation_model")
rel_model.eval()

ID2NER = {1: "PERSON", 2: "ORG", 3: "LOCATION"}



text = input("\nEnter input text:\n").strip()
print("\nINPUT TEXT:")
print(text)



inputs = ner_tokenizer(
    text,
    return_offsets_mapping=True,
    return_tensors="pt",
    truncation=True
)

offsets = inputs.pop("offset_mapping")[0].tolist()
tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
preds = ner_model(**inputs).logits.argmax(dim=-1)[0].tolist()

entities = []
current = ""
current_label = None

for token, (s, e), lab in zip(tokens, offsets, preds):
    if token in ["[CLS]", "[SEP]"] or s == e:
        if current:
            entities.append((current, current_label))
            current, current_label = "", None
        continue

    if lab not in ID2NER:
        if current:
            entities.append((current, current_label))
            current, current_label = "", None
        continue

    word = text[s:e]
    label = ID2NER[lab]

    if token.startswith("##"):
        current += word
    else:
        if current:
            entities.append((current, current_label))
        current = word
        current_label = label

if current:
    entities.append((current, current_label))

entities = [(e, t) for e, t in entities if len(e) > 2]


words = text.split()
if not any(t == "PERSON" for _, t in entities):
    entities.append((words[0], "PERSON"))

print("\nENTITIES:")
print(entities)



main_person = next((e for e, t in entities if t == "PERSON"), words[0])
coref_map = {"He": main_person, "She": main_person}

print("\nCOREFERENCE:")
print(f"Pronoun resolved → {main_person}")



REL_MAP = {
    "works": "works_at",
    "joined": "joined",
    "studies": "studies_at",
    "moved": "lives_in",
    "lives": "lives_in",
    "stays": "lives_in"
}

relations = []
sentences = [s.strip() for s in text.split(".") if s.strip()]

for sent in sentences:
    sl = sent.lower()
    relation = "unknown"
    for k, v in REL_MAP.items():
        if k in sl:
            relation = v
            break

    parts = sent.split()
    subject = coref_map.get(parts[0], parts[0])
    obj = parts[-1]

    relations.append((subject, relation, obj))

print("\nRELATIONS:")
print(relations)



print("\nSCENE GRAPH EDGES:")

G = nx.DiGraph()
for e, t in entities:
    G.add_node(e, label=t)

for s, r, o in relations:
    G.add_edge(s, o, label=r)
    print(f"{s} --{r}--> {o}")

pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
plt.title("Scene Graph Output")
plt.show()

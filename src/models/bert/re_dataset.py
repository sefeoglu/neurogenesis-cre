import os
import json

import torch
from torch.utils.data import Dataset

def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, path):
    """ Write a json file to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def data_preparation(data):
  data_prepared = []
  for item in data:

    sentence = item["sentence"]
    entity1 = item["subject"]
    entity2 = item["object"]
    relation = item["relation"]
    sentence_e1 = sentence.replace(entity1, f"[E1]{entity1}[/E1]")
    sentence_e2 = sentence_e1.replace(entity2, f"[E2]{entity2}[/E2]")
    row = {"sentence": "[CLS] "+sentence_e2, "relation": relation, "e1":entity1, "e2":entity2}
    data_prepared.append(row)
  return data_prepared


class RelationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }




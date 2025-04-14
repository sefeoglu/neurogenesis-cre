import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
random.seed(42)

class BERTEncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout=0.1):
        super(BERTEncoderBlock, self).__init__()
        self.query_fc = nn.Linear(embed_size, embed_size)
        self.key_fc = nn.Linear(embed_size, embed_size)
        self.value_fc = nn.Linear(embed_size, embed_size)
        self.attn_out_fc = nn.Linear(embed_size, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(nn.Linear(embed_size, ff_hidden_size), nn.ReLU(), nn.Linear(ff_hidden_size, embed_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q, K, V = self.query_fc(x), self.key_fc(x), self.value_fc(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.query_fc.out_features ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = self.attn_out_fc(attn_output)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, ff_hidden_size, heads, num_layers, max_length, num_classes, dropout=0.1):
        super(BERTClassifier, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.encoder_layers = nn.ModuleList([BERTEncoderBlock(embed_size, heads, ff_hidden_size, dropout) for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, input_ids):
        seq_length, batch_size = input_ids.size()
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(torch.arange(seq_length, device=input_ids.device).unsqueeze(1).expand(-1, batch_size))
        x = token_embeds + position_embeds

        for encoder in self.encoder_layers:
            x = encoder(x)

        cls_output = x[0]  # Use [CLS] token output
        logits = self.classifier(cls_output)
        return logits
    
class BERTFewRelDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenization (convert words to IDs)
        tokenized = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]

        # Padding to max_length
        if len(tokenized) < self.max_length:
            tokenized += [self.vocab["<PAD>"]] * (self.max_length - len(tokenized))
        else:
            tokenized = tokenized[:self.max_length]

        return torch.tensor(tokenized), torch.tensor(label)
    
class BERTTACREDDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenization (convert words to IDs)
        tokenized = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]

        # Padding to max_length
        if len(tokenized) < self.max_length:
            tokenized += [self.vocab["<PAD>"]] * (self.max_length - len(tokenized))
        else:
            tokenized = tokenized[:self.max_length]

        return torch.tensor(tokenized), torch.tensor(label)
    

def prepare_dataset(texts, labels, vocab, max_length=50):
    dataset = BERTFewRelDataset(texts, labels, vocab, max_length)
    return dataset



def train():
    pass

if __name__ == "__main__":
    pass
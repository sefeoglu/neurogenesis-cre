import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model, test_dataset, batch_size):
    device = get_device()
    model.eval()
    test_preds, test_labels = [], []
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch in test_loader_tqdm:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    test_accuracy = accuracy_score(test_labels, test_preds)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy
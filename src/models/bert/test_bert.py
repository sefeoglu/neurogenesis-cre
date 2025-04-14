import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score

def test_model(model, test_dataset, batch_size):
  model.eval()
  test_preds, test_labels = [], []
  test_loader = DataLoader(test_dataset, batch_size=batch_size)
  test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)
  with torch.no_grad():
      for batch in test_loader_tqdm:
          input_ids = batch['input_ids'].to("cuda")
          attention_mask = batch['attention_mask'].to("cuda")
          labels = batch['label'].to("cuda")
          outputs = model(input_ids=input_ids, attention_mask=attention_mask) # Pass attention_mask to the model

          logits = outputs
          test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())  # Use logits for argmax
          test_labels.extend(labels.cpu().numpy())
  print(test_labels)
  print(test_preds)
  test_accuracy = accuracy_score(test_labels, test_preds)

  print(f"Test Accuracy: {test_accuracy:.4f}")
  return test_accuracy
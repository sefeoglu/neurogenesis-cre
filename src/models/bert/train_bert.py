
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer


def train_model(epochs, batch_size, val_dataset, train_dataset, model, run_id, task_id, patience=3):
    """
    Trains a BERT-based model on a given dataset with validation, early stopping, and best model loading.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and validation.
        val_dataset: Validation dataset.
        train_dataset: Training dataset.
        model: Pretrained model to fine-tune.
        run_id (str): Unique identifier for the training run.
        task_id (str): Task identifier for model saving.
        patience (int): Number of epochs to wait for validation loss improvement before stopping.

    Returns:
        model: The best trained model (based on validation loss).
        tokenizer: The tokenizer associated with the model.
        train_hist: Training history containing loss, accuracy, and learning rate for each epoch.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Adjust learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_hist = []
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement_epochs = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_loader_tqdm:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['label'].to("cuda")

            outputs = model(input_ids, attention_mask)
            logits = outputs

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_loader_tqdm:
                input_ids = batch['input_ids'].to("cuda")
                attention_mask = batch['attention_mask'].to("cuda")
                labels = batch['label'].to("cuda")

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs
                loss = criterion(logits, labels)
                val_loss += loss.item()

                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                val_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)

        scheduler.step(val_loss)

        print(f"  Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Learning Rate = {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': scheduler.optimizer.param_groups[0]['lr']
        }
        train_hist.append(epoch_log)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # Early stopping
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model based on validation loss.")

    print("Training complete.")
    return model, tokenizer, train_hist

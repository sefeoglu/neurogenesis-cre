import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, PreTrainedModel, BertConfig, BertTokenizer
from neuro_genesis import neurogenesis  # Ensure this module is available in your environment
import random
from baseline import BertForSequenceClassification_Neuro
from re_dataset import RelationDataset
from re_dataset import read_json, write_json, data_preparation
import torch.optim as optim

# Set random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

class TaskRouter(nn.Module):
    def __init__(self, input_dim, num_tasks):
        super(TaskRouter, self).__init__()
        self.fc = nn.Linear(input_dim, num_tasks)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)  # Probabilities for each task



# Dynamic Mixture of Tasks Model
class DynamicMixtureOfTasks(nn.Module):
    def __init__(self, input_dim, initial_tasks, neuro_genesis, neuro_phi, num_classes=4):
        super(DynamicMixtureOfTasks, self).__init__()
        self.input_dim = input_dim
        self.neuro_genesis = neuro_genesis
        self.neuro_phi = neuro_phi
        # Task-specific models]
        config = BertConfig.from_pretrained('bert-large-uncased', num_labels=num_classes)
        self.task_models = nn.ModuleList(
            [BertForSequenceClassification_Neuro(config, pretrained_model_name='bert-large-uncased', num_classes=num_classes, use_custom_encoder=True, neuro_genesis=neuro_genesis, phi=neuro_phi) for _ in range(num_tasks)]
        )
        self.router = TaskRouter(input_dim, len(initial_tasks))

    def forward(self, x):
        task_probs = torch.softmax(self.router(x), dim=1)
        task_preds = []
        for i, task_model in enumerate(self.tasks):
            preds = task_model(x) * task_probs[:, i].unsqueeze(1)
            task_preds.append(preds)
        return task_probs, task_preds

    def add_task(self, num_classes):
        config = BertConfig.from_pretrained('bert-large-uncased', num_labels=num_classes)
        new_task_model = BertForSequenceClassification_Neuro(config, pretrained_model_name='bert-large-uncased', num_classes=num_classes, use_custom_encoder=True, neuro_genesis=self.neuro_genesis, phi=self.neuro_phi)
        
        self.tasks.append(new_task_model)

        # Expand router's output
        new_router = nn.Linear(self.input_dim, len(self.tasks))
        with torch.no_grad():
            new_router.weight[:len(self.tasks) - 1] = self.router.weight
            new_router.bias[:len(self.tasks) - 1] = self.router.bias
        self.router = new_router


def retun_new_task_data(run_id, task_id, dataset_path):
    torch.cuda.empty_cache()
    train_data = read_json(f"{dataset_path}/train/run_{run_id}/task{task_id}/train_1.json")
    train_prepared = data_preparation(train_data)
    test_data = read_json(f"{dataset_path}/test/run_{run_id}/task{task_id}/test_1.json")
    test_prepared = data_preparation(test_data)
    val_data = read_json(f"{dataset_path}/train/run_{run_id}/task{task_id}/dev_1.json")
    val_prepared = data_preparation(val_data)
    train_labels = [item['relation'] for item in train_prepared]
    train_sentences = [item['sentence'] for item in train_prepared]
    test_labels = [item['relation'] for item in test_prepared]
    test_sentences = [item['sentence'] for item in test_prepared]
    val_labels = [item['relation'] for item in val_prepared]
    val_sentences = [item['sentence'] for item in val_prepared]
    label_to_int = {label: idx for idx, label in enumerate(set(train_labels))}
    # all_test_sentences.extend(test_sentences)
    # all_test_labels.extend(test_labels)

    # # Convert labels to integers using the pre-calculated mapping
    train_labels = [label_to_int[label] for label in train_labels]
    val_labels = [label_to_int[label] for label in val_labels]
    test_labels = [label_to_int[label] for label in test_labels]

    # test_labels_seen = [label_to_int[label] for label in all_test_labels]

    # Create Dataset and DataLoader
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    train_dataset = RelationDataset(train_sentences, train_labels, tokenizer, max_length=512)
    val_dataset = RelationDataset(val_sentences, val_labels, tokenizer, max_length=512)
    test_dataset = RelationDataset(test_sentences, test_labels, tokenizer, max_length=512)

    return train_dataset, val_dataset, test_dataset, label_to_int, train_labels, val_labels, test_labels, train_sentences, val_sentences, test_sentences


def main(out_dir, dataset_path, neuro_genesis, baseline_name, epoch_list, batch_list, neuro_phi):

    # torch.cuda.empty_cache()
    for run_id in range(2,6):
        results = []

        # all_labels = []
        # all_seen_test_data = []
        # all_test_sentences = []
        # all_test_labels = []

        
        task_id = 1
        train_dataset, val_dataset, test_dataset, label_to_int, train_labels, val_labels, test_labels, train_sentences, val_sentences, test_sentences = retun_new_task_data(run_id, task_id, dataset_path)
        input_dim = 512
        initial_tasks = [4]  # Task 1: 3 classes, Task 2: 5 classes
        model = DynamicMixtureOfTasks(input_dim, initial_tasks, neuro_genesis, neuro_phi,  num_classes=4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Global label offset to ensure no overlapping labels across tasks
        label_offset = sum(initial_tasks)

        # Training Loop with Dynamic Task Addition
        epochs = epoch_list[0]*10
        task_addition_interval = 5

        # Training data for initial tasks
        batch_size = batch_list[0]
        task_specific_labels = train_labels

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            
            task_probs, task_preds = model(train_dataset)
            total_loss = 0

            for task_idx in range(len(model.tasks)):
                task_mask = (train_labels == task_idx).nonzero(as_tuple=True)[0]
                if len(task_mask) > 0:
                    predictions = task_preds[task_idx][task_mask]
                    true_labels = task_specific_labels[task_idx][task_mask]
                    loss = nn.CrossEntropyLoss()(predictions, true_labels)
                    total_loss += loss

            total_loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {total_loss.item()}")

            # Add new task dynamically
            if (epoch + 1) % task_addition_interval == 0:
                
                task_id += 1
                train_dataset, val_dataset, test_dataset, label_to_int, train_labels, val_labels, test_labels, train_sentences, val_sentences, test_sentences = retun_new_task_data(run_id, task_id, dataset_path)
                new_num_classes = len(label_to_int)  # Number of classes for the new task
                print(f"Adding a new task with {new_num_classes} classes at epoch {epoch + 1}")
                model.add_task(new_num_classes)
                
                # Simulate new task data (unique to the new task)
                new_task_data = train_dataset
                new_task_labels = train_labels
                label_offset += new_num_classes  # Update label offset for the next task

                # Train the new task model exclusively
                new_task_optimizer = optim.Adam(model.tasks[-1].parameters(), lr=0.001)
                print(f"{new_task_labels}")
                for new_task_epoch in range(5):  # Train the new task model for a few epochs
                        
                    model.tasks[-1].train()
                    new_task_optimizer.zero_grad()
                    predictions = model.tasks[-1](new_task_data)
                    loss = nn.CrossEntropyLoss()(predictions, new_task_labels - label_offset + new_num_classes)
                    loss.backward()
                    new_task_optimizer.step()
                    print(f"New Task Training Epoch {new_task_epoch}, Loss: {loss.item()}")

         # del model
        del tokenizer
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    #drive folde=EMNLP-neurogenesis

    # phi = ['cosine','performer', 'linear', 'truncated_performer', 'positive_cosine', 'dima_sin']
    phi = ['performer']
    neuro_genesis = True
    epoch_list = [5]
    batch_list = [16]
    for neuro_genesis_type_phi in phi:
      baseline_name = "bert_large_performer_neurogenesis"
      dataset_path = "/content/tacred/final"
      output_dir = f"drive/MyDrive/EMNLP-neurogenesis/neurogenesis_results_{neuro_genesis_type_phi}"

      main(output_dir, dataset_path, neuro_genesis,baseline_name, epoch_list, batch_list, neuro_genesis_type_phi)

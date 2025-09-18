import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

# Set random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

class DynamicMixtureOfTasks(nn.Module):
    def __init__(self, input_dim, num_classes, initial_num_tasks):
        super(DynamicMixtureOfTasks, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_tasks = initial_num_tasks

        # Router: Determines task probabilities
        self.router = nn.Linear(input_dim, initial_num_tasks)

        # Task-specific models
        self.task_models = nn.ModuleList(
            [TaskModel(input_dim, num_classes) for _ in range(initial_num_tasks)]
        )

    def forward(self, x):
        task_probs = torch.softmax(self.router(x), dim=1)  # Task probabilities
        task_preds = []
        for task_idx, task_model in enumerate(self.task_models):
            preds = task_model(x) * task_probs[:, task_idx].unsqueeze(1)
            task_preds.append(preds)
        return task_probs, task_preds

    def add_task(self):
        # Add a new task-specific model
        new_task_model = TaskModel(self.input_dim, self.num_classes)
        self.task_models.append(new_task_model)

        # Expand the router's output dimension
        new_router = nn.Linear(self.input_dim, self.num_tasks + 1)
        with torch.no_grad():
            new_router.weight[:self.num_tasks] = self.router.weight
            new_router.bias[:self.num_tasks] = self.router.bias
        self.router = new_router

        # Update the number of tasks
        self.num_tasks += 1

# Single Task Model
class TaskModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TaskModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Parameters
input_dim = 100
num_classes = 3
initial_num_tasks = 2

# Initialize model
model = DynamicMixtureOfTasks(input_dim, num_classes, initial_num_tasks)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example Training Data
batch_size = 10
data = torch.randn(batch_size, input_dim)
task_labels = torch.randint(0, initial_num_tasks, (batch_size,))
task_specific_labels = [torch.randint(0, num_classes, (batch_size,)) for _ in range(initial_num_tasks)]

# Training Loop with Task Addition
epochs = 10
task_addition_interval = 3  # Add a new task every 3 epochs

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    task_probs, task_preds = model(data)

    total_loss = 0
    for task_idx in range(model.num_tasks):
        task_mask = task_labels == task_idx
        if task_mask.any():
            predictions = task_preds[task_idx][task_mask]
            true_labels = task_specific_labels[task_idx][task_mask]
            loss = nn.CrossEntropyLoss()(predictions, true_labels)
            total_loss += loss

    total_loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {total_loss.item()}")

    # Dynamically add a new task at intervals
    if (epoch + 1) % task_addition_interval == 0:
        print(f"Adding a new task at epoch {epoch + 1}")
        model.add_task()
        # Simulate new task data (replace with actual data in practice)
        new_task_labels = torch.randint(0, num_classes, (batch_size,))
        task_specific_labels.append(new_task_labels)
        print(f"Total tasks now: {model.num_tasks}")

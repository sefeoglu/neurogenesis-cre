import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, PreTrainedModel, BertConfig
from neuro_genesis import neurogenesis  # Ensure this module is available in your environment
import random
from baseline import BertForSequenceClassification_Neuro
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


# Mixture of Tasks model
class MixtureOfTasks(nn.Module):
    def __init__(self, input_dim, num_classes, num_tasks, neuro_genesis=True, neuro_phi='performer'):
        super(MixtureOfTasks, self).__init__()
        self.router = TaskRouter(input_dim, num_tasks)
        config = BertConfig.from_pretrained('bert-large-uncased', num_labels=num_classes)
        self.task_models = nn.ModuleList(
            [BertForSequenceClassification_Neuro(config, pretrained_model_name='bert-large-uncased', num_classes=num_classes, use_custom_encoder=True, neuro_genesis=neuro_genesis, phi=neuro_phi) for _ in range(num_tasks)]
        )

    def forward(self, x):
        task_probs = self.router(x)  # Predict task probabilities
        task_preds = []
        for task_idx, task_model in enumerate(self.task_models):
            preds = task_model(x) * task_probs[:, task_idx].unsqueeze(1)
            task_preds.append(preds)
        return task_probs, task_preds  # Return probabilities and task-specific predictions

    def predict(self, x):
        task_probs, task_preds = self.forward(x)
        selected_task = torch.argmax(task_probs, dim=1)  # Select most probable task
        final_preds = []
        for i, task_idx in enumerate(selected_task):
            final_preds.append(task_preds[task_idx][i])  # Get predictions for the selected task
        return torch.stack(final_preds)
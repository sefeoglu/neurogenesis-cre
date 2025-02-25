"""RE model"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

class REModel(nn.Module):
    """
    Relation Extraction model
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x
    def loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true)

    def accuracy(self, y_pred, y_true):
        return torch.sum(torch.argmax(y_pred, dim=1) == y_true).item() / len(y_true)

    def optimizer(self, lr):
        return optim.Adam(self.parameters(), lr=lr) 

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def predict(self, x):
        return torch.argmax(self(x), dim=1)

    def predict_proba(self, x):
        return self(x)

    def predict_proba_numpy(self, x):
        return self(x).detach().numpy()

    
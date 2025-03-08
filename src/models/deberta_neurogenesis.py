import torch
import torch.nn as nn

class NeuronLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, output_dim)

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x
    def astrocyte_process(self):
        """ apply astrocyte activation function """
        pass
    def microglia_process(self):
        """ apply microglia activation function """
        pass
    def oligodendrocyte_process(self):
        """ apply oligodendrocyte activation function """
        pass
    def save(self, path):
        torch.save(self.state_dict(), path)
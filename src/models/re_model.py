"""RE model for Encoders using Neurogenesis Astrocyte mechnaism"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from models.neurogenesis import ProliferationLayer

class AddAndNormalize(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        Initializes the Add & Normalize module.
        
        Args:
            d_model (int): Dimensionality of the input and output.
            eps (float): A small value added to the denominator for numerical stability in LayerNorm.
        """
        super(AddAndNormalize, self).__init__()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
    
    def forward(self, x, sublayer_output):
        """
        Forward pass for Add & Normalize.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            sublayer_output (Tensor): Output from the sublayer (e.g., attention or FFN) of the same shape.
        
        Returns:
            Tensor: Output tensor after residual connection and layer normalization.
        """
        # Add residual connection
        residual = x + sublayer_output
        
        # Apply layer normalization
        normalized_output = self.layer_norm(residual)
        
        return normalized_output


class REModel(nn.Module):
    """
    Relation Extraction model
    """


    def __init__(self, input_dim, output_dim, attention_layer, custom_layer):
        super().__init__()
    
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.attention_layer = attention_layer
        self.custom_layer = custom_layer
        self.ProliferationLayer = ProliferationLayer(self.attention_layer, self.custom_layer)
        self.add_and_norm_layer = AddAndNormalize(input_dim)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(input_dim, output_dim)
        

    def forward(self, x):
        
        x_astro_ps_high_m, x_astro_ps_low_m, x_attention_normalizations = self.ProliferationLayer(x)
        out_norm = self.add_and_norm_layer(x, x_attention_normalizations)

        fc_output = self.fc(out_norm)
        out_norm =self.add_and_norm_layer = self.add_and_norm_layer(x, fc_output)

        return out_norm

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

    
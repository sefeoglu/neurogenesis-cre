import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, PreTrainedModel
from neuro_genesis import neurogenesis  # Ensure this module is available in your environment
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

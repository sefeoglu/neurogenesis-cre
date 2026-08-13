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



class CustomBERTEncoderBlock(nn.Module):
    def __init__(self, embed_size, ff_hidden_size, dropout=0.1, neurogenesis=True, phi='performer'):
        
        super(CustomBERTEncoderBlock, self).__init__()
        self.query_fc = nn.Linear(embed_size, embed_size)
        self.key_fc = nn.Linear(embed_size, embed_size)
        self.value_fc = nn.Linear(embed_size, embed_size)
        self.attn_out_fc = nn.Linear(embed_size, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.neurogenesis = neurogenesis
        self.phi = phi

    def forward(self, x):
        query = self.query_fc(x)
        key = self.key_fc(x)
        value = self.value_fc(x)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        # Apply neurogenesis only if enabled
        if self.neurogenesis:
            low = neurogenesis(512, query, key, 4, phi=self.phi)
            # # Convert low and high to PyTorch tensors before applying softmax
            low = torch.tensor(low, device=x.device, dtype=torch.float32)  # Ensure correct data type
            attn_weights = F.softmax(low, dim=-1)
        else:
            attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x, key, query, value


class BertForSequenceClassification_Neuro(PreTrainedModel):
    def __init__(
        self,
        config,
        pretrained_model_name='bert-large-uncased',
        num_classes=8,
        ff_hidden_size=2048,
        dropout=0.1,
        use_custom_encoder=True,
        neuro_genesis=True,
        phi='performer',
        num_experts=4,
    ):

        super(BertForSequenceClassification_Neuro, self).__init__(config)
        self.bert = BertModel.from_pretrained(pretrained_model_name, config=config)
        self.use_custom_encoder = use_custom_encoder
        self.num_labels = num_classes
        self.num_experts = max(1, num_experts)
        embed_size = self.bert.config.hidden_size

        if use_custom_encoder:
            self.custom_encoder = CustomBERTEncoderBlock(embed_size, ff_hidden_size, dropout, neurogenesis=neurogenesis, phi=phi)

        self.classifier = nn.Linear(embed_size, num_classes)
        self.router = nn.Linear(embed_size, self.num_experts)
        self.experts = nn.ModuleList([nn.Linear(embed_size, num_classes) for _ in range(self.num_experts)])
        self.dropout = nn.Dropout(dropout)

    def _mixture_of_experts(self, hidden_states):
        router_logits = self.router(hidden_states)
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_logits = torch.stack([expert(hidden_states) for expert in self.experts], dim=0)
        return torch.sum(router_probs.unsqueeze(-1) * expert_logits, dim=0), router_probs

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_router_probs=False):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output

        if self.use_custom_encoder:
            sequence_output, key, query, value = self.custom_encoder(sequence_output)
            pooled_output = sequence_output[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        if self.num_experts > 1:
            logits, router_probs = self._mixture_of_experts(pooled_output)
            if return_router_probs:
                return logits, router_probs
            return logits

        logits = self.classifier(pooled_output)
        if return_router_probs:
            router_probs = torch.ones(logits.size(0), 1, device=logits.device)
            return logits, router_probs
        return logits

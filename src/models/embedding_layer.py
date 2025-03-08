

import torch.nn as nn
from gat_layer import GraphAttentionLayer

import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, model_name="microsoft/deberta-base"):
        super(EmbeddingLayer, self).__init__()
        self.embedding_model = model_name
        self.load_tokenizer()
        self.load_embedding()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.embedding

    
    def load_data(self, dataset_id):

        dataset = load_dataset(dataset_id)
        # print("Dataset loaded: ", dataset['train'][0]['data'])
        return dataset

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)


    def load_embedding(self):
        # model_name = "microsoft/deberta-base"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.emb_model = AutoModel.from_pretrained(self.embedding_model)
        self.emb_model.to(self.device)


    def tokenize(self, text):

        encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        return encoded_input

    def get_embeddings(self, text):


        inputs = self.tokenize(text)

        # Get model outputs
        with torch.no_grad():
            outputs = self.emb_model(**inputs)

        # Get embeddings from the last hidden state
        # Shape: (batch_size, sequence_length, hidden_size)

        token_embeddings = outputs.last_hidden_state
        # # Convert token IDs back to tokens for reference
        # tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

        return token_embeddings.squeeze(0)


    
    # def prepare_task_data(self, dataset, run_id, task_id):
    #     """
    #     Prepare data for training
    #     """
    #     task_data = dataset['train'][run_id]['data'][task_id]['samples']
    #     return task_data

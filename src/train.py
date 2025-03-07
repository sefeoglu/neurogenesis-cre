import os
import sys
import configparser
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
# sys.path.append(PREFIX_PATH)

from transformers import BertTokenizer, BertModel
from models.gat_layer import GraphAttentionLayer
from models.re_model import REModel

class Trainer(object):
    def __init__(self, config_path):

        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self.data_set = self.config["DATA"]["dataset"]
        self.train_path = self.config["DATA"]["train_path"]
        self.test_path = self.config["TEST"]["test_path"]
        self.embedding_model = self.config["MODEL"]["embedding_model"]
        print("Data set: ", self.data_set)
        print("Train path: ", self.train_path)

        self.load_tokenizer()
        self.load_embedding()

        #hyperparameters
        self.EPOCHS = self.config["HYPERPARAMETERS"]["epochs"]
        self.BATCH_SIZE = self.config["HYPERPARAMETERS"]["batch_size"]
        self.lr = self.config["HYPERPARAMETERS"]["lr"]
        self.alpha = self.config["HYPERPARAMETERS"]["alpha"]
        self.dropout = self.config["HYPERPARAMETERS"]["dropout"]
        # device check
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # define model instance
        self.model = REModel(input_dim=768, output_dim=2)
        self.model.to(self.device)

    def data_cleaning(self, text):
        """
        Clean data and remove unwanted characters

        """
        return text.replace('( )', '').rstrip().lstrip()

    def load_data(self, dataset_id):

        dataset = load_dataset(dataset_id)
        # print("Dataset loaded: ", dataset['train'][0]['data'])
        return dataset

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)


    def load_embedding(self):
        # model_name = "microsoft/deberta-base"
        
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

    def input_prepation(self, item):
        """
        Prepare input for training and compute model embeddings
        """
        # print("Data: ", item)
        input = dict()
        input['sentence'] = self.data_cleaning(item['sentence'])
        input['entity1'] = item['entity1']
        input['entity2'] = item['entity2']
        input['relation'] = item['relation']
        input['sentence_embedding'] = self.get_embeddings(self.tokenize(item['sentence']))
        input['entity1_embedding'] = self.get_embeddings(self.tokenize(item['entity1']))
        input['entity2_embedding'] = self.get_embeddings(self.tokenize(item['entity2']))
        input['relation_embedding'] = self.get_embeddings(self.tokenize(item['relation']))
        
        return input
    
    def prepare_task_data(self, dataset, run_id, task_id):
        """
        Prepare data for training
        """
        task_data = dataset['train'][run_id]['data'][task_id]['samples']
        return task_data


    def get_clean_data(self, dataset):
        """
        Clean data and remove unwanted characters
        """
        train_data_emb = []
        
        for i, data in enumerate(dataset):

            item = data['data']

            _input = self.input_prepation(item)

            train_data_emb.append(_input)
   
        return train_data_emb

    
    def generate_batches(samples, batch_size):

        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def train(self, run_id=0, task_id=0):

        # load data
        print("Loading data...")
        self.train_data = self.load_data(self.train_path)
        self.task_data = self.prepare_task_data(self.train_data, run_id, task_id)

        # print("Task data: ", self.task_data)
        print("Data loaded successfully")
        print("prepare data for training...")

        # tokenize data
        # get embeddings
        train_emb = self.get_clean_data(self.task_data)
        # save train get_embeddings
        #graph attention network for entity pair embeddings and word embeddings
        # save train data
        np.save(PREFIX_PATH + "data/train_emb.npy", train_emb)
        # model training

        print("Training model...")

        for epoch in tqdm(range(1, self.EPOCHS+1)):
            for batch_samples in generate_batches(train_emb, self.BATCH_SIZE):
                # train model
                with torch.gradient():
                ## TODO ##
                # custom training function
                    pass

        # evaluate
        # save model
    


    def evaluate(self):
        pass

    def save_model(self):
        pass




if __name__ == '__main__':
    config_path = PREFIX_PATH + "config.ini"
    print("Training model with config: ", config_path)
    print("Loading config file...")
    print(config_path)

    trainer = Trainer(config_path)

    trainer.train()







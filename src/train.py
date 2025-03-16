import os
import sys
import configparser
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
# sys.path.append(PREFIX_PATH)

from transformers import BertTokenizer, BertModel
from models.gat_layer import GraphAttentionLayer
from models.re_model import REModel
from models.embedding_layer import EmbeddingLayer

from data_preparation.prepare_dependency_matrix import prepare_dependency_matrix

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
       
        #hyperparameters
        self.EPOCHS = self.config["HYPERPARAMETERS"]["epochs"]
        self.BATCH_SIZE = self.config["HYPERPARAMETERS"]["batch_size"]
        self.lr = self.config["HYPERPARAMETERS"]["lr"]
        self.alpha = self.config["HYPERPARAMETERS"]["alpha"]
        self.dropout = self.config["HYPERPARAMETERS"]["dropout"]
        # device check
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #define embedding layer
        self.embedding_layer = EmbeddingLayer(self.embedding_model)
        # define model instance
        # self.model = REModel(input_dim=768, output_dim=2)
        # self.model.to(self.device)


    def data_cleaning(self, text):
        """
        Clean data and remove unwanted characters

        """
        return text.replace('( )', '').rstrip().lstrip()
    def get_gat_entity_emb(self, text, entity1, entity2):
        """
        Get entity embeddings using GAT
        """
        # print(text)
        # print(entity1)
        # print(entity2)
        entity_single_1 = entity1.replace(' ', '_').replace('-','_').replace('.','').replace('/','_').rstrip().lstrip()
        entity_single_2 = entity2.replace(' ', '_').replace('-','_').replace('.','').replace('/','_').rstrip().lstrip()
        text = text.replace(entity1, entity_single_1).replace(entity2, entity_single_2)

        words, matrix = prepare_dependency_matrix(text)
        
        word_emb = [np.array(self.embedding_layer.get_embeddings(word), dtype=np.float32).reshape(-1) for word in words]

        max_len = max(len(emb) for emb in word_emb)
        word_emb = [np.pad(emb, (0, max_len - len(emb)), 'constant') for emb in word_emb]
      
        matrix = torch.tensor(matrix, dtype=torch.float32)
        word_emb = np.array(word_emb, dtype=np.float32)  # Specify dtype for word_emb
        word_emb = torch.tensor(word_emb, dtype=torch.float32)
        # print(words)
        ent1_index= words.index(entity_single_1)
        ent2_index= words.index(entity_single_2)

        self.gat_layer = GraphAttentionLayer(in_features=word_emb.shape[1], out_features=word_emb.shape[1], dropout=0.6, alpha=0.2, concat=True)


        with torch.no_grad():
            h_primes = self.gat_layer(word_emb, matrix)
        
        entity_1_prime = h_primes[ent1_index]
        entity_2_prime = h_primes[ent2_index]
        return h_primes, entity_1_prime, entity_2_prime

        
    def input_prepation(self, item):
        """
        Prepare input for training and compute model embeddings
        """
        input = dict()

        input['sentence'] = self.data_cleaning(item['sentence'])
        input['entity1'] = item['subject']
        input['entity2'] = item['object']
        input['relation'] = item['relation']
        input['sentence_embedding'] = self.embedding_layer.get_embeddings(item['sentence'])

        input['entity1_embedding'] = self.embedding_layer.get_embeddings(item['subject'])
        input['entity2_embedding'] = self.embedding_layer.get_embeddings(item['object'])
        input['relation_embedding'] = self.embedding_layer.get_embeddings(item['relation'])
        h_primes, entity_1_prime, entity_2_prime = self.get_gat_entity_emb(item['sentence'], item['subject'], item['object'])
        input['gat_entity_1'] = entity_1_prime
        input['gat_entity_2'] =  entity_2_prime
        input['gat_entity_prime'] = h_primes
        # print(gat_emb)
        return input

    def get_data_embeddings(self, dataset):
            """
            Clean data and remove unwanted characters
            """
            task = read_json('tacred_5way.json')[0]['run_1'][0]['task1']
            train_data_emb = []
            # print(dataset)
            dataset = [ data for data in dataset if data['relation'] != 'no_relation']
            dataset = [ item for item in dataset if item['relation'] in task[:2]]
            print(len(dataset))
           
            
            for i, item in enumerate(dataset):
                _input = self.input_prepation(item)

                # Move tensors in the dictionary to CPU individually
                for key, value in _input.items():
                  if isinstance(value, torch.Tensor):
                    _input[key] = value.to('cpu')
                train_data_emb.append(_input) # Append the modified dictionary outside the inner loop
            return train_data_emb # Moved the return statement outside the main loop

    def load_data(self, dataset_id):

        dataset = load_dataset(dataset_id)
        # print("Dataset loaded: ", dataset['train'][0]['data'])
        return dataset

    def generate_batches(self,samples, batch_size):

        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def train(self, run_id=0, task_id=0):

        # load data
        print("Loading data...")
        self.data = self.load_data(self.train_path)
        # self.task_data = self.prepare_task_data(self.train_data, run_id, task_id)

        # print("Task data: ", self.task_data)
        print("Data loaded successfully")
        print("prepare data for training...")

        # tokenize data
        # get embeddings
        self.train_data = self.data['train']
        # print("Train data: ", self.train_data)
        train_emb = self.get_data_embeddings(self.train_data)
        # self.val_data = self.data['validation']
        # val_emb = self.get_data_embeddings(self.val_data)
        # self.test_data = self.data['test']
        # test_emb = self.get_data_embeddings(self.test_data)
        # print("Train data embeddings: ", train_emb)
        # save train get_embeddings
        #graph attention network for entity pair embeddings and word embeddings
        # save train data
        np.save("train_emb.npy", train_emb)
        # model training

        # print("Training model...")

        # for epoch in tqdm(range(1, self.EPOCHS+1)):
        #     for batch_samples in self.generate_batches(train_emb, self.BATCH_SIZE):
        #         # train model
        #         with torch.gradient():
        #         ## TODO ##
        #         # custom training function
        #             pass

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
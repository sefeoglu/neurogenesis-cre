import os
import sys
import configparser
from datasets import load_dataset

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
# sys.path.append(PREFIX_PATH)
from transformers import BertTokenizer, BertModel

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


    def data_cleaning(self):
        """
        Clean data and remove unwanted characters
        """
        pass

    def load_data(self, dataset_id):

        dataset = load_dataset(dataset_id)
        # print("Dataset loaded: ", dataset['train'][0]['data'])
        return dataset

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.embedding_model)


    def load_embedding(self):
        self.bert_model = BertModel.from_pretrained(self.embedding_model)
        

    def tokenize(self, text):

        encoded_input = self.tokenizer(text, return_tensors='pt')

        return encoded_input

    def get_embeddings(self, encoded_input):

        output_embedding = self.bert_model(**encoded_input)
        return output_embedding

    def input_prepation(self, data):
        """
        Prepare input for training and compute model embeddings
        """
        input = dict()
        input['sentence'] = data['sentence']
        input['entity1'] = data['entity1']
        input['entity2'] = data['entity2']
        input['relation'] = data['relation']
        input['sentence_embedding'] = self.get_embeddings(self.tokenize(data['sentence']))
        input['entity1_embedding'] = self.get_embeddings(self.tokenize(data['entity1']))
        input['entity2_embedding'] = self.get_embeddings(self.tokenize(data['entity2']))
        input['relation_embedding'] = self.get_embeddings(self.tokenize(data['relation']))
        
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
        for data in dataset:
            # print("Data: ", d)
            data = data['data']

            # sentence = data['sentence']
            # entity1 = data['entity1']
            # entity2 = data['entity2']
            # relation_type = data['relation']
            # print("entity1: ", entity1)
            # print("entity2: ", entity2)
            # print("relation: ", relation_type)
            # print("sentence: ", sentence)

            _input = self.input_prepation(data)
            print("Input: ", _input)
            train_data_emb.append(_input)
        return train_data_emb

    def train(self):
        # load data
        self.train_data = self.load_data(self.train_path)
        self.task_data = self.prepare_task_data(self.train_data, 0, 0)
        # print("Task data: ", self.task_data)
        train_emb = self.get_clean_data(self.task_data)


        # tokenize data


        # get embeddings
        # train model
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







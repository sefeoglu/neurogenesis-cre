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
        self.tokenizer = BertTokenizer.from_pretrained(self.embedding_model)


    def load_embedding(self):
        self.bert_model = BertModel.from_pretrained(self.embedding_model)
        

    def tokenize(self, text):

        encoded_input = self.tokenizer(text, return_tensors='pt')

        return encoded_input

    def get_embeddings(self, encoded_input):

        output_embedding = self.bert_model(**encoded_input)
        return output_embedding

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
        np.save(PREFIX_PATH + "data/train_emb.npy", train_emb)
        
        print("Train data: ", train_emb)
        
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







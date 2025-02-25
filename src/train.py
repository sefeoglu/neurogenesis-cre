import os
import sys
import configparser


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
        load_tokenizer()
        load_embedding()
        self.train_data = load_data(self.train_path)

    def load_data(self, dataset_id):
        dataset = load_dataset(dataset_id)
    return dataset

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.embedding_model)


    def load_embedding(self):
        self.bert_model = BertModel.from_pretrained(self.embedding_model)
        

    def tokenize(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        return encoded_input

    def get_embeddings(self, encoded_input):
        output_embedding = self.model(**encoded_input)
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

        

    def train(self):
        # tokenize data

        # get embeddings
        # train model
        # evaluate
        # save model
        pass


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







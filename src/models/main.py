from transformers import BertTokenizer, BertConfig
import torch
from re_dataset import RelationDataset, data_preparation, read_json, write_json
from train_bert import train_model
from test_bert import test_model
from baseline import BertForSequenceClassification_Neuro
# Set random seeds for reproducibility
import random
import numpy as np
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

def main(out_dir, dataset_path, neuro_genesis, baseline_name,  batch_list, epoch_list):

    
    # torch.cuda.empty_cache()
    for run_id in range(1,6):
        results = []

        # all_labels = []
        # all_seen_test_data = []
        # all_test_sentences = []
        # all_test_labels = []

        for task_id in range(1, 11):

            train_data = read_json(f"{dataset_path}/train/run_{run_id}/task{task_id}/train_1.json")
            train_prepared = data_preparation(train_data)
            test_data = read_json(f"{dataset_path}/test/run_{run_id}/task{task_id}/test_1.json")
            test_prepared = data_preparation(test_data)
            val_data = read_json(f"{dataset_path}/train/run_{run_id}/task{task_id}/dev_1.json")
            val_prepared = data_preparation(val_data)
            # all_labels.extend([item['relation'] for item in train_prepared])
            # Create unique mapping for all labels
            

            train_labels = [item['relation'] for item in train_prepared]
            train_sentences = [item['sentence'] for item in train_prepared]
            test_labels = [item['relation'] for item in test_prepared]
            test_sentences = [item['sentence'] for item in test_prepared]
            val_labels = [item['relation'] for item in val_prepared]
            val_sentences = [item['sentence'] for item in val_prepared]
            label_to_int = {label: idx for idx, label in enumerate(set(train_labels))}
            # all_test_sentences.extend(test_sentences)
            # all_test_labels.extend(test_labels)

            # # Convert labels to integers using the pre-calculated mapping
            train_labels = [label_to_int[label] for label in train_labels]
            val_labels = [label_to_int[label] for label in val_labels]
            test_labels = [label_to_int[label] for label in test_labels]

            # test_labels_seen = [label_to_int[label] for label in all_test_labels]

            # Create Dataset and DataLoader
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


                # Load the configuration first
            config = BertConfig.from_pretrained('bert-large-uncased', num_labels=len(label_to_int))
            # Then pass it to the model constructor
            model = BertForSequenceClassification_Neuro(config, pretrained_model_name='bert-large-uncased', num_classes=len(label_to_int), use_custom_encoder=True, neuro_genesis=neuro_genesis)
            # else:

            #     #The old model should have the total number of labels encountered till the last task.
            #     old_labels = model.num_labels
            #     print(f"Old labels: {old_labels}")
            #     model = increment_class_labels(model, new_num_labels=len(label_to_int)) #Use total number of labels for current task

            train_dataset = RelationDataset(train_sentences, train_labels, tokenizer, max_length=512)
            val_dataset = RelationDataset(val_sentences, val_labels, tokenizer, max_length=512)

            test_dataset = RelationDataset(test_sentences, test_labels, tokenizer, max_length=512)
            # all_seen_test_data = RelationDataset(all_test_sentences, test_labels_seen, tokenizer, max_length=512)

            for i, epoch in enumerate(epoch_list):

                for batch in batch_list:
                    model, tokenizer, train_hist = train_model(epoch, batch, val_dataset, train_dataset, model, run_id, task_id)

                    write_json(train_hist, f"{out_dir}/train_hist_{run_id}_{task_id}.json")
                    model_name_hf = f"{baseline_name}_{run_id}_{task_id}"
                    # base_model_hf = f"Sefika/bert_large_baseline_{run_id}_{task_id}"
                    model.push_to_hub(model_name_hf, private=True)
                    test_acc = test_model(model, test_dataset, batch)
                    # test_seen_acc = test_model(model, all_seen_test_data, batch)

                    print(f"Epoch-------------{epoch}=={i}")
                    result = {"run_id":run_id, "task_id": task_id, "epoch": epoch, "batch_size": batch, "test_acc": test_acc}
                    results.append(result)
                    write_json(results, f"{out_dir}/results_{run_id}.json")

                    torch.cuda.empty_cache()
                # del model
        del tokenizer
        del model

if __name__ == "__main__":
    output_dir = "./neurogenesis_results_low"
    neuro_genesis = True
    epoch_list = [1]
    batch_list = [16]
    baseline_name = "bert_large_baseline_neurogenesis"
    dataset_path = "/content/tacred/final"
    main(output_dir, dataset_path, neuro_genesis,baseline_name, epoch_list, batch_list)

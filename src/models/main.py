import argparse
import random
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertTokenizer

from baseline import BertForSequenceClassification_Neuro
from re_dataset import RelationDataset, data_preparation, read_json, write_json
from test_bert import test_model
from train_bert import train_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_task_split(dataset_path, run_id, task_id):
    train_data = read_json(f"{dataset_path}/train/run_{run_id}/task{task_id}/train_1.json")
    val_data = read_json(f"{dataset_path}/train/run_{run_id}/task{task_id}/dev_1.json")
    test_data = read_json(f"{dataset_path}/test/run_{run_id}/task{task_id}/test_1.json")

    return (
        data_preparation(train_data),
        data_preparation(val_data),
        data_preparation(test_data),
    )


def prepare_labels(records):
    labels = [item["relation"] for item in records]
    sentences = [item["sentence"] for item in records]
    label_to_int = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    encoded_labels = [label_to_int[label] for label in labels]
    return sentences, encoded_labels, label_to_int


def build_model(num_labels, neuro_genesis, neuro_phi, model_name="bert-large-uncased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
    model = BertForSequenceClassification_Neuro(
        config,
        pretrained_model_name=model_name,
        num_classes=num_labels,
        use_custom_encoder=True,
        neuro_genesis=neuro_genesis,
        phi=neuro_phi,
    )
    return tokenizer, model


def main(out_dir, dataset_path, neuro_genesis, baseline_name, epoch_list, batch_list, neuro_phi, model_name="bert-large-uncased"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for run_id in range(1, 6):
        results = []

        for task_id in range(1, 11):
            train_prepared, val_prepared, test_prepared = load_task_split(dataset_path, run_id, task_id)

            train_sentences, train_labels, label_to_int = prepare_labels(train_prepared)
            val_sentences, val_labels, _ = prepare_labels(val_prepared)
            test_sentences, test_labels, _ = prepare_labels(test_prepared)

            tokenizer, model = build_model(len(label_to_int), neuro_genesis, neuro_phi, model_name=model_name)

            train_dataset = RelationDataset(train_sentences, train_labels, tokenizer, max_length=512)
            val_dataset = RelationDataset(val_sentences, val_labels, tokenizer, max_length=512)
            test_dataset = RelationDataset(test_sentences, test_labels, tokenizer, max_length=512)

            for epoch in epoch_list:
                for batch in batch_list:
                    model, tokenizer, train_hist = train_model(epoch, batch, val_dataset, train_dataset, model, run_id, task_id)
                    write_json(train_hist, str(out_dir / f"train_hist_{run_id}_{task_id}.json"))

                    model_name_hf = f"{baseline_name}_{run_id}_{task_id}"
                    model.push_to_hub(model_name_hf, private=True)

                    test_acc = test_model(model, test_dataset, batch)
                    result = {
                        "run_id": run_id,
                        "task_id": task_id,
                        "epoch": epoch,
                        "batch_size": batch,
                        "test_acc": test_acc,
                    }
                    results.append(result)
                    write_json(results, str(out_dir / f"results_{run_id}.json"))

                    torch.cuda.empty_cache()

        del tokenizer
        del model


def parse_args():
    parser = argparse.ArgumentParser(description="Train relation extraction models with neurogenesis.")
    parser.add_argument("--dataset-path", type=str, default="./data/tacred/final")
    parser.add_argument("--output-dir", type=str, default="./results/neurogenesis")
    parser.add_argument("--phi", type=str, default="performer", choices=["performer", "cosine", "linear", "truncated_performer", "positive_cosine", "dima_sin"])
    parser.add_argument("--baseline-name", type=str, default="bert_large_performer_neurogenesis")
    parser.add_argument("--model-name", type=str, default="bert-large-uncased", help="Base Hugging Face model to load for the BERT encoder")
    parser.add_argument("--epochs", type=int, nargs="+", default=[1])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--neuro-genesis", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    main(
        out_dir=args.output_dir,
        dataset_path=args.dataset_path,
        neuro_genesis=args.neuro_genesis,
        baseline_name=args.baseline_name,
        epoch_list=args.epochs,
        batch_list=args.batch_sizes,
        neuro_phi=args.phi,
        model_name=args.model_name,
    )

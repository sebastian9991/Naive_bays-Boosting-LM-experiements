import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EvalPrediction, Trainer, TrainingArguments)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


def get_data() -> Dataset:
    """Get data hugging face Dataset format."""
    ds = load_dataset("google-research-datasets/go_emotions", "raw", split = "train" )
    # ds_test = load_dataset("google-research-datasets/go_emotions", "raw", split = "test")
    return ds


def get_tokenization(dataset: Dataset):
    """
    Split the data into appropriate formats X, y
    preprocess text into tokenized format.
    """
    dataset = dataset.remove_columns(
        [
            "id",
            "author",
            "subreddit",
            "link_id",
            "parent_id",
            "created_utc",
            "rater_id",
            "example_very_unclear",
        ]
    )
    dataset = remove_multi_labels(dataset)
    labels = [label for label in dataset.features.keys() if label not in ["text", "id"]]
    encoding_dataset = dataset.map(
        lambda row: preprocess_data(row, labels), batched=False
    )
    return encoding_dataset, labels


def preprocess_data(dataset: Dataset, labels):
    """Preprocess according to tutorial: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=AFWlSsbZaRLc"""
    text = dataset["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True)
    labels_batch = {k: dataset[k] for k in dataset.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def remove_multi_labels(dataset: Dataset):
    """Texts with more than one label are dropped for simplicity."""
    ohe_columns = [col for col in dataset.column_names if col != "text"]
    dataset = dataset.filter(lambda row: sum(row[col] for col in ohe_columns) > 1)
    return dataset


def main() -> None:
    data = get_data()
    data_encoded, labels = get_tokenization(data)
    data_encoded.set_format("torch") #Create training, validation, test sets
    print(data_encoded[0])
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    batch_size = 128
    training_args = TrainingArguments(
        f"bert-finetuned-sem_eval-reddit",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model,
        args = training_args,
        train_dataset = data_encoded['train'],
        eval_dataset = data_encoded['test'],
        tokenizer = tokenizer,
        compute_metrics = compute_metrics

    )

    trainer.train()


if __name__ == "__main__":
    main()

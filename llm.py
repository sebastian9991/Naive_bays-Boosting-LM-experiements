import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EvalPrediction, Trainer, TrainingArguments)
import os
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


def get_data() -> Dataset:
    """Get data hugging face Dataset format."""
    ds = load_dataset("google-research-datasets/go_emotions", "raw", split="train")
    # ds_test = load_dataset("google-research-datasets/go_emotions", "raw", split = "test")
    ds = ds.remove_columns(
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
    ds = ds.train_test_split(test_size=0.2)
    return ds


data = get_data()
labels = [
    label for label in data["train"].features.keys() if label not in ["text", "id"]
]


def get_tokenization(dataset: Dataset):
    """
    Split the data into appropriate formats X, y
    preprocess text into tokenized format.
    """
    dataset["train"] = remove_multi_labels(dataset["train"])
    dataset["test"] = remove_multi_labels(dataset["test"])
    encoding_dataset = dataset.map(
        preprocess_data, batched=True, remove_columns=dataset["train"].column_names
    )
    return encoding_dataset


def preprocess_data(dataset: Dataset):
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
    # Set environment variable for PyTorch memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # Clear GPU memory
    torch.cuda.empty_cache()

    # Reset CUDA memory
    torch.cuda.reset_peak_memory_stats()

    data_encoded = get_tokenization(data)
    data_encoded.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}


    print("Loading Model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    print("Model loaded.")
    batch_size = 32
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
        fp16=True,
        dataloader_pin_memory=True,
    )

    print("Training arguments configured")
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=data_encoded["train"],
        eval_dataset=data_encoded["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete")

    print(torch.cuda.memory_summary(device=torch.device("cuda")))


if __name__ == "__main__":
    main()

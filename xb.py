import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tabulate import tabulate
from wikipedia2vec import Wikipedia2Vec

MODEL_FILE = "./enwiki_20180420_win10_100d.pkl"

wiki2vec = Wikipedia2Vec.load(MODEL_FILE)
vector_dim = wiki2vec.get_word_vector("the").shape[0]


def get_data():
    splits = {
        "train": "simplified/train-00000-of-00001.parquet",
        "validation": "simplified/validation-00000-of-00001.parquet",
        "test": "simplified/test-00000-of-00001.parquet",
    }
    train = pd.read_parquet(
        "hf://datasets/google-research-datasets/go_emotions/" + splits["train"]
    ).drop(columns="id")
    val = pd.read_parquet(
        "hf://datasets/google-research-datasets/go_emotions/" + splits["validation"]
    ).drop(columns="id")
    test = pd.read_parquet(
        "hf://datasets/google-research-datasets/go_emotions/" + splits["test"]
    ).drop(columns=["id"])

    train = remove_multi_labels(train)
    val = remove_multi_labels(val)
    test = remove_multi_labels(test)
    return train, val, test


def vectorize_get_X_y(dataframe):
    """
    Split the data into appropriate formats X, y
    preprocess text into vectorized format.
    """
    vec = TfidfVectorizer()
    count_matrix = vec.fit_transform(dataframe["text"])
    # Get vector input X, and OHE label y
    y = dataframe.drop(columns=["text"])
    y = [e[1].iloc[0][0] for e in y.iterrows()]
    return count_matrix, np.asarray(y)

def vectorize_count_X_y(dataframe):
    """
    Split the data into appropriate formats X, y
    preprocess text into vectorized format.
    """
    vec = CountVectorizer()
    count_matrix = vec.fit_transform(dataframe["text"])
    # Get vector input X, and OHE label y
    y = dataframe.drop(columns=["text"])
    y = [e[1].iloc[0][0] for e in y.iterrows()]
    return count_matrix, np.asarray(y)

def word_embeddings_get_X_y(dataframe):

    def embed_text(text):
        tokens = text.split()
        embeddings = []
        for token in tokens:
            try:
                embeddings.append(wiki2vec.get_word_vector(token))
            except KeyError:
                continue

                
        if embeddings:
            return np.mean(embeddings, axis=0)  # Average it for a sentence embedding
        else:
            return np.zeros(vector_dim)

    embedding_matrix = np.vstack(dataframe["text"].apply(embed_text))
    y = dataframe.drop(columns=["text"])
    y = [e[1].iloc[0][0] for e in y.iterrows()]
    return embedding_matrix, np.asarray(y)


def remove_multi_labels(dataframe):
    """Texts with more than one label are dropped for simplicity."""
    dataframe = dataframe[dataframe["labels"].apply(lambda x: len(x) == 1)]
    return dataframe


def display_evaluation(y_true, y_pred, filename: str):
    """Display and calculate evalution metrics."""
    # Create a ohe_columns by scratch due to removing the columns in the read data method
    # Assume order is correct given in paper
    ohe_columns = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Create a DataFrame for the confusion matrix for better readability
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f"True {cat}" for cat in ohe_columns],
        columns=[f"Pred {cat}" for cat in ohe_columns],
    )

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (XGB): {filename}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.savefig(
        f"./figs/CM_xgb_{filename}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

    accuracy = accuracy_score(y_true, y_pred)
    table_acc = [[accuracy]]
    print(tabulate(table_acc, headers=["Accuracy"], tablefmt="pretty"))

    report = classification_report(
        y_true, y_pred, target_names=ohe_columns, output_dict=True
    )
    class_metrics = []
    for cat in ohe_columns:
        cat_name = f"{cat}"
        precision = report[str(cat)]["precision"]
        recall = report[str(cat)]["recall"]
        f1_score = report[str(cat)]["f1-score"]
        class_metrics.append([cat_name, precision, recall, f1_score])

    headers = ["class", "precision", "Recall", "F1-score"]
    accuracy_row = ["Final Accuracy", "-", "-", accuracy]
    class_metrics.append(accuracy_row)
    table_class = tabulate(
        class_metrics, headers=headers, tablefmt="grid", floatfmt=".2f"
    )
    print(table_class)

    # Create plot for the table
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed
    ax.axis("tight")
    ax.axis("off")

    # Generate table using matlplotlib
    table = ax.table(
        cellText=class_metrics,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))
    # Highlight accuracy row
    num_rows = len(class_metrics)
    for i in range(len(headers)):
        table[(num_rows - 1, i)].set_facecolor("#f0f0f0")
    fig.subplots_adjust(top=0.82)
    fig.subplots_adjust(right=0.696)

    # Save or display the figure
    plt.title(f"Classification Report (XGB): {filename}", fontsize=14, weight="bold")
    plt.savefig(
        f"./figs/classification_report_table_xgb{filename}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def main():
    """With guidance from: https://www.kaggle.com/code/stuarthallows/using-xgboost-with-scikit-learn"""
    train, val, test = get_data()
    # Train
    x, y = word_embeddings_get_X_y(train)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    xgb_model = xgb.XGBClassifier(objective="multi:softprob")
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_train)
    display_evaluation(y_train, y_pred, filename="train_Word_embeddings")
    # Test
    y_pred_test = xgb_model.predict(x_test)
    display_evaluation(y_test, y_pred_test, filename="test_Word_embeddings")
    ##TFID
    x, y = vectorize_get_X_y(train)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    xgb_model = xgb.XGBClassifier(objective="multi:softprob")
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_train)
    display_evaluation(y_train, y_pred, filename="train_tfid")
    # Test
    y_pred_test = xgb_model.predict(x_test)
    display_evaluation(y_test, y_pred_test, filename="test_tfid")

    #Count Vec
    x, y = vectorize_count_X_y(train)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    xgb_model = xgb.XGBClassifier(objective="multi:softprob")
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_train)
    display_evaluation(y_train, y_pred, filename="train_count")
    # Test
    y_pred_test = xgb_model.predict(x_test)
    display_evaluation(y_test, y_pred_test, filename="test_count")


if __name__ == "__main__":
    main()

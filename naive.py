import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tabulate import tabulate

NUM_CLASSES = 28


def to_OHE(y):
    """Convert class array to OHE."""

    one_hot = np.zeros((len(y), NUM_CLASSES), dtype=int)
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


class NaiveBayes:
    def __init__(self, bayesian=False) -> None:
        self.bayesian = bayesian

    def fit(self, X, y):
        self.calculate_priors(y)
        self.calculate_feature_conditional(X, y)

    def predict(self, X):
        """Assumes X is 1xD vector"""
        prob_joint_c_x = self.calculate_joint(X)
        self.prob_joint_c_x = prob_joint_c_x

        # arg-max to find final classification
        y_pred = []
        pred_range = []
        for key, value in prob_joint_c_x.items():
            y_pred.append(np.argmax(value))
            pred_range.append(value)

        norm_pred_range = [
            range / (np.sum(range)) for range in pred_range
        ]  # Normalize to conditional probabilites

        return np.asarray(y_pred), np.asarray(norm_pred_range)

    def evaluate_acc(self, y_true, y_pred):
        """Cross-entropy"""
        y_true_ohe = to_OHE(y_true)
        assert y_pred.shape == y_true_ohe.shape
        assert y_true_ohe.shape[0] != 0
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true_ohe * np.log(y_pred)) / y_true_ohe.shape[0]
        return loss

    def evaluate_acc_confusion(self, y_true, y_pred):
        """sci-kit diagonal sum"""
        matrix = confusion_matrix(y_true, y_pred)
        return matrix.trace() / matrix.sum()

    def calculate_joint(self, X):
        assert isinstance(X, csr_matrix)
        N, D = X.shape

        prob_x_given_c = {
            key: value.toarray() for key, value in self.prob_xd_given_class.items()
        }
        prob_x_not_given_c = {
            key: 1 - value.toarray() for key, value in self.prob_xd_given_class.items()
        }

        # Convert to dense for faster operations
        X_dense = X.toarray()

        prob_joint_c_x = {}

        for n in range(N):
            row = X_dense[n, :]
            joint_probs = []

            for c, prior in self.prior_probs.items():
                # Compute log joint probability
                product = np.prod(
                    (row * prob_x_given_c[c]) + ((1 - row) * prob_x_not_given_c[c])
                )
                joint = prior * product
                joint_probs.append(joint)  # Convert back to probability

            prob_joint_c_x[n] = joint_probs

        return prob_joint_c_x

    def calculate_priors(self, y):
        ##learn prior
        N = y.shape[0]
        self.prior_counts = {}
        for c in y:
            if c in self.prior_counts:
                self.prior_counts[c] += 1
            else:
                self.prior_counts[c] = 1

        # Sort the counts for argmax usage
        self.prior_counts = dict(sorted(self.prior_counts.items()))
        prior_probs = {}
        for key, values in self.prior_counts.items():
            prior_probs[key] = values / N

        self.prior_probs = prior_probs

    def calculate_feature_conditional(self, X, y):
        # Assume for X feature = 1
        prob_xd_given_class = {}
        for x_row, c in zip(X, y):
            if c in prob_xd_given_class:
                prob_xd_given_class[c] = np.add(prob_xd_given_class[c], x_row)
            else:
                prob_xd_given_class[c] = x_row.copy()

        # normalize
        for key, value in prob_xd_given_class.items():
            prob_xd_given_class[key] = value / self.prior_counts[key]

        self.prob_xd_given_class = prob_xd_given_class


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
    vec = CountVectorizer()
    count_matrix = vec.fit_transform(dataframe["text"])
    # Get vector input X, and OHE label y
    y = dataframe.drop(columns=["text"])
    y = [e[1].iloc[0][0] for e in y.iterrows()]
    return count_matrix, np.asarray(y)


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
    plt.title(f"Confusion Matrix: {filename}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(
        f"./figs/CM_naive_{filename}.png",
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
    fig.subplots_adjust(top=0.82)
    fig.subplots_adjust(right=0.696)

    # Save or display the figure
    plt.title(f"Classification Report (Naive): {filename}", fontsize=12, weight="bold")
    plt.savefig(
        f"./figs/classification_report_table_naive_{filename}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def main():

    nb = NaiveBayes()
    # Train
    print("Training on train dataset")
    train, val, test = get_data()
    x, y = vectorize_get_X_y(train)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )
    nb.fit(x_train, y_train)
    y_train_predict, y_train_predict_range = nb.predict(x_train)
    loss = nb.evaluate_acc(y_train, y_train_predict_range)
    acc = nb.evaluate_acc_confusion(y_train, y_train_predict)
    table = [[loss, acc]]
    print(
        tabulate(
            table,
            headers=["Cross-Entropy Loss (Train)", "Accuracy (Train)"],
            tablefmt="pretty",
        )
    )
    display_evaluation(y_train, y_train_predict, filename="train")
    # Test
    y_test_predict, y_test_predict_range = nb.predict(x_test)
    loss = nb.evaluate_acc(y_test, y_test_predict_range)
    acc = nb.evaluate_acc_confusion(y_test, y_test_predict)
    table = [[loss, acc]]
    print(
        tabulate(
            table,
            headers=["Cross-Entropy Loss (Test)", "Accuracy (Test)"],
            tablefmt="pretty",
        )
    )
    display_evaluation(y_test, y_test_predict, filename="test")
    return 0


main()

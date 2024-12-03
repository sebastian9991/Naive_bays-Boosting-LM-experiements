import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


##CLASS##
class NaiveBayes:
    def __init__(self, bayesian=False) -> None:
        self.bayesian = bayesian
        pass

    def fit(self, X, y):
        self.calculate_priors(y)
        self.calculate_feature_conditional(X, y)

    def predict(self, X):
        """Assumes X is 1xD vector"""
        prob_joint_c_x = self.calculate_joint(X)

        sum = 0
        for key, value in prob_joint_c_x.items():
            sum += value


        ##Normalize
        prob_x_given_c = {}
        for key, value in prob_joint_c_x.items():
            prob_x_given_c[key] = prob_joint_c_x[key] / sum

        for key, value in prob_x_given_c.items():
            print(f"Key:{key}")
            print(f"probability: {value}")
        return prob_x_given_c

    def evaluate_acc(self, y_true, y_pred):
        pass

    def calculate_joint(self, X):
        assert isinstance(X, csr_matrix)
        row = X.getrow(0)
        prob_joint_c_x = {}
        for c in self.prior_probs:
            product = 1
            for idx, d in enumerate(range(0, row.shape[1])):
                value = row[(0, d)]
                print(f"value: {value}")

                if value == 1:
                    print(f"Accessed for 1: {self.prob_xd_given_class[c][(0, idx)]}")
                    product = product * self.prob_xd_given_class[c][(0, idx)]
                else:
                    print(f"Accessed for 0: {1 - self.prob_xd_given_class[c][(0, idx)]}")
                    product = product * (1 - self.prob_xd_given_class[c][(0, idx)])
            prob_joint_c_x[c] = self.prior_probs[c]*product
        return prob_joint_c_x


    def calculate_priors(self, y):
        ##learn prior
        N = y.shape[0]
        self.prior_counts = {}
        for c in y:
            if f"{c}" in self.prior_counts:
                self.prior_counts[f"{c}"] += 1
            else:
                self.prior_counts[f"{c}"] = 1

        prior_probs = {}
        for key, values in self.prior_counts.items():
            prior_probs[key] = values / N

        self.prior_probs = prior_probs

    def calculate_feature_conditional(self, X, y):
        # Assume for X feature = 1
        prob_xd_given_class = {}
        for x_row, c in zip(X, y):
            if f"{c}" in prob_xd_given_class:
                prob_xd_given_class[f"{c}"] = np.add(prob_xd_given_class[f"{c}"], x_row)
            else:
                prob_xd_given_class[f"{c}"] = x_row.copy()

        # normalize
        for key, value in prob_xd_given_class.items():
            prob_xd_given_class[key] = value / self.prior_counts[key]

        for key, value in prob_xd_given_class.items():
            print(f"Key:{key}")
            print(f"Value matrix: {value}")

        self.prob_xd_given_class = prob_xd_given_class


def get_data():
    df = pd.read_parquet(
        "hf://datasets/google-research-datasets/go_emotions/raw/train-00000-of-00001.parquet"
    )
    df = df.drop(
        columns=[
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
    emotion_columns = df.columns[1:]
    df = df[df[emotion_columns].sum(axis=1) <= 1]
    ohe_columns = df.drop(columns=["text"]).columns
    df["noEmotion"] = (df[ohe_columns].sum(axis=1) == 0).astype(
        int
    )  # No classification cases
    return df


def vectorize_get_X_y(dataframe):
    """
    Split the data into appropriate formats X, y
    preprocess text into vectorized format.
    """
    vec = CountVectorizer()
    tfidf_matrix = vec.fit_transform(dataframe["text"])
    # Get vector input X, and OHE label y
    y = dataframe.drop(columns=["text"])
    ohe_columns = y.columns
    y["class_labels"] = np.argmax(y[ohe_columns].values, axis=1)
    y = y.drop(columns=ohe_columns)
    return tfidf_matrix, y 


def main():
    df = get_data()
    X, y = vectorize_get_X_y(df)

    return 0


main()

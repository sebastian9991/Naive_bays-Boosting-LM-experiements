import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


##CLASS##
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


        #arg-max to find final classification
        y_pred = []
        for key, value in prob_joint_c_x.items():
            y_pred.append(np.argmax(value))

        return y_pred





    def evaluate_acc(self, y_true, y_pred):
        pass

    def calculate_joint(self, X):
        assert isinstance(X, csr_matrix)
        N = X.shape[0]
        prob_joint_c_x = {}
        for n in range(0, N):
            row = X.getrow(n)
            joint_probs = []
            iterator = dict(sorted(self.prior_probs.items(), key=lambda item: item[0]))
            for c, prob in iterator.items():
                print(f"Calculating joint under predict: {c}")
                product = 1
                for idx, d in enumerate(range(0, row.shape[1])):
                    value = row[(0, d)]

                    if value == 1:
                        product = product * self.prob_xd_given_class[c][(0, idx)]
                    else:
                        product = product * (1 - self.prob_xd_given_class[c][(0, idx)])

                joint_probs.append(self.prior_probs[c] * product)

            prob_joint_c_x[n] = joint_probs
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
            print(f"Calculating class prob: {c}")
            if f"{c}" in prob_xd_given_class:
                prob_xd_given_class[f"{c}"] = np.add(prob_xd_given_class[f"{c}"], x_row)
            else:
                prob_xd_given_class[f"{c}"] = x_row.copy()

        # normalize
        for key, value in prob_xd_given_class.items():
            prob_xd_given_class[key] = value / self.prior_counts[key]

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
    count_matrix = vec.fit_transform(dataframe["text"])
    # Get vector input X, and OHE label y
    y = dataframe.drop(columns=["text"])
    ohe_columns = y.columns
    y["class_labels"] = np.argmax(y[ohe_columns].values, axis=1)
    y = y.drop(columns=ohe_columns)
    y = y.to_numpy()
    return count_matrix, y


def main():
    # X = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
    # y = ['class1', 'class2', 'class1']
    # y = np.asarray(y)
    nb = NaiveBayes()
    # nb.calculate_priors(y)
    # nb.calculate_feature_conditional(X, y)


    X = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
    y = [0, 1, 0]
    y = np.asarray(y)
    nb.fit(X, y)
    nb.predict(X)

    return 0


main()

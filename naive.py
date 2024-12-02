import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


##CLASS##
class NaiveBayes:
    def __init__(self) -> None:
        pass


    def fit(self, X, y):
        self.prior_probs = self.calculate_priors(y)







    def predict(self, X):
        pass


    def evaluate_acc(self, y_true, y_pred):
        pass

    def calculate_priors(self, y):
        ##learn prior
        N = y.shape[0]
        self.prior_counts = {}
        for c in y:
            if f'{c}' in self.prior_counts:
                self.prior_counts[f'{c}'] += 1
            else:
                self.prior_counts[f'{c}'] = 1

        
        prior_probs = {}
        for key, values in self.prior_counts.items():
            prior_probs[key] = values / N

        return prior_probs
    def calculate_feature_conditional(self, X, y):
        #Assume for X feature = 1
        prob_xd_given_class = {}
        for x_row, c in zip(X, y):
            if f'{c}' in prob_xd_given_class:
                prob_xd_given_class[f'{c}'] = np.add(prob_xd_given_class[f'{c}'],  x_row)
            else:
                prob_xd_given_class[f'{c}'] = x_row.copy()

        #normalize
        for key, value in prob_xd_given_class.items():
            prob_xd_given_class[key] = value / self.prior_counts[key]

        for key, value in prob_xd_given_class.items():
            print(f'Key is {key}')
            print(value)
            print(self.prior_counts[key])











            


        


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
    return tfidf_matrix, dataframe.drop(columns=["text"])


def main():
    df = get_data()
    X, y = vectorize_get_X_y(df)

    return 0

main()

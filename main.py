import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import seaborn as sns


def get_data():
    """
    Gets the data from google-research-datasets, drops uneeded columns, and removes OHE with more than one occurence of 1.
    """
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
    df["noEmotion"] = (df[ohe_columns].sum(axis=1) == 0).astype(int) #No classification cases
    return df


def vectorize_get_X_y(dataframe):
    """
    Split the data into appropriate formats X, y
    preprocess text into vectorized format.
    """
    vec = TfidfVectorizer()
    tfidf_matrix = vec.fit_transform(dataframe["text"])
    # Get vector input X, and OHE label y
    return tfidf_matrix, dataframe.drop(columns=["text"])


def main():
    """With guidance from: https://www.kaggle.com/code/stuarthallows/using-xgboost-with-scikit-learn"""
    df = get_data()
    X, y = vectorize_get_X_y(df)
    ohe_columns = y.columns
    y["class_labels"] = np.argmax(y[ohe_columns].values, axis=1)
    y = y.drop(columns=ohe_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ##XGBoost

    xgb_model = xgb.XGBClassifier(objective="multi:softprob")
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_train)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_train, y_pred)

    # Create a DataFrame for the confusion matrix for better readability
    conf_matrix_df = pd.DataFrame(
        conf_matrix, 
        index=[f"True {cat}" for cat in ohe_columns],
        columns=[f"Pred {cat}" for cat in ohe_columns]
    )

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()



    accuracy = accuracy_score(y_train, y_pred)
    table = [[accuracy]]
    print(tabulate(table, headers = ["Accuracy"], tablefmt = "pretty"))


    report = classification_report(y_train, y_pred, target_names = ohe_columns, output_dict = True)
    class_metrics = []
    for cat in ohe_columns:
        cat_name = f"{cat}"
        precision = report[str(cat)]["precision"]
        recall = report[str(cat)]["recall"]
        f1_score = report[str(cat)]["f1-score"]
        class_metrics.append([cat_name, precision, recall, f1_score])

    headers = ["class", "precision", "Recall", "F1-score"]
    table_class = tabulate(class_metrics, headers=headers, tablefmt= "grid", floatfmt = ".2f")
    print(table_class)



if __name__ == "__main__":
    main()

import os

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

from tools.quadratic_metric_kappa import quadratic_kappa
from tools.quadratic_metric_kappa import list_of_class_values_from_df
from tools.file_worker import read_data
from tools.file_worker import write_submission
from tools.make_model import ModelMaker
from tools.prepare_data import prepare_data
from tools.make_model import prepare_hash_train_and_test_kaggle


def make_forecast(train, train_labels, test):
    hyperparams = {
        "n_neighbors": 7,
        "weights": "distance",
        "algorithm": "auto",
        "leaf_size": 15,
        "p": 2,
        "metric": "minkowski",
    }

    train_dataset = prepare_data(train, train_labels)

    #x_train, x_test, y_train, y_test, test_ist_ids = prepare_train_and_test(train_dataset)

    x_train, x_test_hash, y_train, test = prepare_hash_train_and_test_kaggle(train_dataset, test)

    model = ModelMaker(KNeighborsClassifier, hyperparams, x_train, y_train, x_test_hash)
    prediction = model.predict()
    write_submission(test['installation_id'], prediction, "submission.csv")
    # print(quadratic_kappa(y_test, prediction, 4))
    return prediction


if __name__ == "__main__":
    input_path = os.path.join("..", "Data")
    output_path = os.path.join("..", "Prediction")
    files = ["train.csv", "train_labels.csv", "test.csv"]

    data = read_data(files, input_path)

    make_forecast(data["train.csv"], data["train_labels.csv"], data["test.csv"])

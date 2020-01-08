import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from tools.data_preprocessing import prepare_train_dataset_due_to_train_labels
from tools.data_preprocessing import prepare_train_dataset_and_test
from tools.data_preprocessing import process_data
from tools.model_maker import ModelMaker
from tools.file_worker import write_submission
from tools.file_worker import read_data
from tools.data_preprocessing import split_train_and_test
from tools.quadratic_metric_kappa import quadratic_kappa


if __name__ == "__main__":

    # choose the mode of working
    isKaggle = True

    # read data
    # input_path = os.path.join('kaggle', 'input', 'data-science-bowl-2019')
    input_path = os.path.join("..", "Data")
    files = ["train.csv", "train_labels.csv", "test.csv", "sample_submission.csv"]

    data = read_data(files, input_path)

    test = data["test.csv"]
    train = data["train.csv"]
    train_labels = data["train_labels.csv"]
    sample_submission = data["sample_submission.csv"]

    # prepare data for prediction
    train = prepare_train_dataset_due_to_train_labels(train, train_labels)
    train, test = prepare_train_dataset_and_test(train, test)

    # data processing
    X_train, y_train, X_test, y_test = process_data(train, test)

    # test mode
    if not isKaggle:
        y_train = pd.DataFrame(y_train, columns=["accuracy_group"])
        train = pd.concat([X_train, y_train], sort=False, axis=1)
        X_train, X_test, y_train, y_test = split_train_and_test(train)

    # set hyperparams
    GBC_hyperparams = {"random_state": 42, "n_estimators": 100}

    # make prediction
    model = ModelMaker(
        GradientBoostingClassifier, GBC_hyperparams, X_train, y_train, X_test
    )
    prediction = model.predict()

    # count metric if it is test mode
    if not isKaggle:
        print(quadratic_kappa(y_test, prediction, 4))

    write_submission(
        sample_submission["installation_id"].tolist(), prediction, "submission.csv"
    )

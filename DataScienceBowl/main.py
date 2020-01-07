import os
import pandas as pd
import numpy as np

import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tools.data_preprocessing import prepare_train_dataset_due_to_train_labels
from tools.data_preprocessing import prepare_train_dataset_and_test
from tools.data_preprocessing import find_unique_values
from tools.data_preprocessing import process_data
from tools.data_preprocessing import make_hashes
from tools.data_preprocessing import encode_data
from tools.data_preprocessing import make_win_codes
from tools.data_preprocessing import process_assessments
from tools.model_maker import ModelMaker
from tools.file_worker import write_submission
from tools.file_worker import read_data


if __name__ == "__main__":
#read data
    input_path = os.path.join('kaggle', 'input', 'data-science-bowl-2019')
    files = ["train.csv", "train_labels.csv", "test.csv", "sample_submission.csv"]

    data = read_data(files, input_path)

    test = data["test.csv"]
    train = data["train.csv"]
    train_labels = data["train_labels.csv"]
    sample_submission = data["sample_submission.csv"]

# prepare data for prediction
    train, test = prepare_train_dataset_and_test(train, test)
    train = prepare_train_dataset_due_to_train_labels(train, train_labels)

    X_train, y_train, X_test = process_data(train, test)

# set hyperparams
    GBC_hyperparams = {
        'random_state': 42,
        'n_estimators': 100
    }

# make prediction
    model = ModelMaker(GradientBoostingClassifier, GBC_hyperparams, X_train, y_train, X_test)
    prediction = model.predict()
    write_submission(sample_submission['installation_id'].tolist(), prediction, "submission.csv")

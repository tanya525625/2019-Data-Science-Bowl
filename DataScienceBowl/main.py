import os

from sklearn.ensemble import GradientBoostingClassifier

from tools.data_preprocessing import prepare_train_dataset_due_to_train_labels
from tools.data_preprocessing import prepare_train_dataset_and_test
from tools.data_preprocessing import process_data
from tools.model_maker import ModelMaker
from tools.file_worker import write_submission
from tools.file_worker import read_data
from tools.data_preprocessing import split_train_and_test


if __name__ == "__main__":
    # choose the mode of working
    isKaggle = False
    # read data
    #input_path = os.path.join('kaggle', 'input', 'data-science-bowl-2019')
    input_path = os.path.join('DataScienceBowl', 'Data')
    files = ["train.csv", "train_labels.csv", "test.csv", "sample_submission.csv"]

    data = read_data(files, input_path)

    test = data["test.csv"]
    train = data["train.csv"]
    train_labels = data["train_labels.csv"]
    sample_submission = data["sample_submission.csv"]

    # prepare data for prediction
    train, test = prepare_train_dataset_and_test(train, test)
    train = prepare_train_dataset_due_to_train_labels(train, train_labels)

    if not isKaggle:
        X_train, X_test, y_train, y_test = split_train_and_test(train)
    else:
        X_train, y_train, X_test = process_data(train, test)

    # set hyperparams
    GBC_hyperparams = {
        'random_state': 42,
        'n_estimators': 100
    }

    print(X_train)
    print(y_test)
    # make prediction
    model = ModelMaker(GradientBoostingClassifier, GBC_hyperparams, X_train, y_train, X_test)
    prediction = model.predict()
    if not isKaggle:
        
    write_submission(sample_submission['installation_id'].tolist(), prediction, "submission.csv")

import os

from sklearn.neighbors import KNeighborsClassifier

from tools.quadratic_metric_kappa import quadratic_kappa
from tools.file_worker import read_data
from tools.file_worker import write_submission
from tools.make_model import ModelMaker
from tools.prepare_data import prepare_train_data
from tools.prepare_data import prepare_test_data
from tools.make_model import prepare_hash_train_and_test_kaggle
from tools.make_model import prepare_train_and_test


def make_forecast(train, train_labels, test_dataset):
    hyperparams = {
        'n_neighbors': 7,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 10,
        'p': 2,
        'metric': 'minkowski'
    }

    train_dataset = prepare_train_data(train, train_labels)
    test_dataset = prepare_test_data(test_dataset)

    x_train, x_test, y_train, y_test, test_ist_ids = prepare_train_and_test(train_dataset)

    model = ModelMaker(KNeighborsClassifier, hyperparams, x_train, y_train, x_test)
    prediction = model.predict()
    write_submission(test_ist_ids["installation_id"].tolist(), prediction, "submission.csv")
    
    # x_train, x_test_hash, y_train, test_dataset_ids = prepare_hash_train_and_test_kaggle(train_dataset, test_dataset)

    # model = ModelMaker(KNeighborsClassifier, hyperparams, x_train, y_train, x_test_hash)
    # prediction = model.predict()
    # write_submission(test_dataset_ids.tolist(), prediction, "submission.csv")

    print(quadratic_kappa(y_test, prediction, 4))
    return prediction


if __name__ == "__main__":
    input_path = os.path.join("..", "Data")
    output_path = os.path.join("..", "Prediction")
    files = ["train.csv", "train_labels.csv", "test.csv"]

    data = read_data(files, input_path)

    make_forecast(data["train.csv"], data["train_labels.csv"], data["test.csv"])

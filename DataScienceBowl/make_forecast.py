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
from tools.prepare_data import prepare_data, prepare_test_for_us


def make_forecast(train, train_labels, test_dataset):
    hyperparams = {
        'n_neighbors': 3,
        'weights': 'uniform',
        'algorithm': 'auto',
        'leaf_size': 3,
        'p': 2,
        'metric': 'minkowski'
    }
    # train_dataset = prepare_train_data(train, train_labels)
    # test_dataset = prepare_test_data(test_dataset)

    isKaggle = False
    test_dataset, train_dataset, test_inst_id_not_enc = prepare_data(test_dataset, train, train_labels, isKaggle)

    #if isKaggle==True: df["accuracy_group"] = NaN
    if isKaggle:
        test_dataset.drop("accuracy_group", inplace=True, axis=1)

    x_train, x_test_hash, y_train = prepare_hash_train_and_test_kaggle(train_dataset, test_dataset)
    if not isKaggle:
        real_value = x_test_hash["accuracy_group"]
        x_test_hash = x_test_hash.drop("accuracy_group", axis=1)

    model = ModelMaker(KNeighborsClassifier, hyperparams, x_train, y_train, x_test_hash)
    prediction = model.predict()
    # print(test_inst_id_not_enc)
    write_submission(test_inst_id_not_enc, prediction, "submission.csv")

    if not isKaggle:
        print(quadratic_kappa(real_value, prediction, 4))
    return 1


if __name__ == "__main__":
    input_path = os.path.join("..", "Data")
    output_path = os.path.join("..", "Prediction")
    files = ["train.csv", "train_labels.csv", "test.csv"]

    data = read_data(files, input_path)

    make_forecast(data["train.csv"], data["train_labels.csv"], data["test.csv"])

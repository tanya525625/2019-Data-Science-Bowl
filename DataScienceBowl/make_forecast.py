import os

from sklearn.neighbors import KNeighborsClassifier

from tools.quadratic_metric_kappa import quadratic_kappa
from tools.quadratic_metric_kappa import list_of_class_values_from_file
from tools.file_worker import read_data
from tools.null_processing import drop_nones
from tools.file_worker import write_submission
from tools.make_model import ModelMaker


def make_forecast(data: dict):
    train_dataset = drop_nones(data)

    # write dataset if it's necessary
    # fw.write_df(train_dataset, "new_train.csv")

    hyperparams = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
    }

    model = ModelMaker(KNeighborsClassifier, hyperparams, train_dataset)
    prediction = model.predict()
    write_submission(model.test_ist_ids, prediction, "submission.csv")


if __name__ == "__main__":
    input_path = os.path.join("..", "Data")
    output_path = os.path.join("..", "Prediction")

    actuals_path = os.path.join(input_path, "sample_submission.csv")
    preds_path = os.path.join(output_path, "predictions.csv")

    files = ("sample_submission.csv", "test.csv",
             "train.csv", "train_labels.csv")
    data = read_data(files, input_path)

    make_forecast(data)

    actuals = list_of_class_values_from_file(actuals_path)
    preds = list_of_class_values_from_file(preds_path)

    evaluation = quadratic_kappa(actuals, preds, 4)

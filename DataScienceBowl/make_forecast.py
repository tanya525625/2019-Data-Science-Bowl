import os

from tools.quadratic_metric_kappa import quadratic_kappa, list_of_class_values
from tools.file_worker import read_data


def make_forecast(data):
    pass


if __name__ == "__main__":
    input_path = os.path.join("..", "Data")
    output_path = os.path.join("..", "Prediction")

    actuals_path = os.path.join(input_path, "sample_submission.csv")
    preds_path = os.path.join(output_path, "predictions.csv")

    files = ("sample_submission.csv", "test.csv", "train.csv", "train_labels.csv")
    data = read_data(files, input_path)

    make_forecast(data)

    actuals = list_of_class_values(actuals_path)
    preds = list_of_class_values(preds_path)

    evaluation = quadratic_kappa(actuals, preds, 4)

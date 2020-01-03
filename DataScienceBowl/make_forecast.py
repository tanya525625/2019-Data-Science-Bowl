import os

from sklearn.neighbors import KNeighborsClassifier

from tools.quadratic_metric_kappa import quadratic_kappa
from tools.quadratic_metric_kappa import list_of_class_values_from_file
from tools.file_worker import read_data, FileWorker
from tools.null_processing import cacheable_drop_nones
from tools.file_worker import write_submission
from tools.make_model import ModelMaker


def make_forecast(data_path: str, data_files: list):
    fw = FileWorker
    train_dataset = cacheable_drop_nones(data_path, data_files)
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
    print(quadratic_kappa(model.y_test, prediction, 4))
    return prediction


if __name__ == "__main__":
    input_path = os.path.join("./", "Data")
    output_path = os.path.join(".", "Prediction")

    actuals_path = os.path.join(input_path, "sample_submission.csv")
    preds_path = os.path.join(output_path, "predictions.csv")

    files = ["sample_submission.csv", "test.csv",
             "train.csv", "train_labels.csv"]
    # data = read_data(files, input_path)
    # data['train.csv'] = data['train_part.csv']
    # data['train_part.csv'] = None
    make_forecast(input_path, files)
    #
    # actuals = list_of_class_values_from_file(actuals_path)
    # preds = list_of_class_values_from_file(preds_path)
    #
    # evaluation = quadratic_kappa(actuals, preds, 4)

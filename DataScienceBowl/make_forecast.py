import os

# from tools.quadratic_metric_kappa import quadratic_kappa
from tools.file_worker import read_data
from tools.null_processing import drop_nones


def make_forecast(data: dict):
    drop_nones(data)

    # write dataset if it's necessary
    # fw = FileWorker()
    # fw.write_df(train_dataset, "new_train.csv")


if __name__ == "__main__":
    input_path = os.path.join("..", "Data")
    output_path = os.path.join("..", "Prediction")
    files = ("sample_submission.csv", "test.csv",
             "train.csv", "train_labels.csv")

    # actuals_filename = "sample_submission.csv"
    # preds_filename = "predictions.csv"

    data = read_data(files, input_path)

    make_forecast(data)

    # evaluation = quadratic_kappa(input_path, output_path,
    #                              actuals_filename, preds_filename, 4)
    # print('Evaluation =', evaluation)

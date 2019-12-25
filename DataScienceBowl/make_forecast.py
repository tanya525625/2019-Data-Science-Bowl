import os
from quadratic_metric_kappa import quadratic_kappa
from tools.file_worker import read_data

def make_forecast(data):
    pass


if __name__ == "__main__":
    input_path = os.path.join("..", "Data")
    output_path = os.path.join("..", "Prediction")
    actuals_filename = "sample_submission.csv"
    preds_filename = "predictions.csv"
    files = ("sample_submission.csv", "test.csv", "train.csv", "train_labels.csv")
    data = read_data(files, input_path)
    
    make_forecast(data)

    evaluation = quadratic_kappa(input_path, output_path, actuals_filename, preds_filename, 4)
    print('Evaluation =', evaluation)
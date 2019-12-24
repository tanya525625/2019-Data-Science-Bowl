import os
from DataScienceBowl.tools.file_worker import read_data


def make_forecast(data):
    pass


if __name__ == "__main__":
    input_path = os.path.join("..", "Data")
    output_path = os.path.join("..", "Prediction")
    files = ("sample_submission.csv", "test.csv", "train.csv", "train_labels.csv")
    data = read_data(files, input_path)
    make_forecast(data)

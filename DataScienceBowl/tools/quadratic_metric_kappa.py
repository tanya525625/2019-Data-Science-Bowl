import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


def quadratic_kappa(actuals, preds, N=4):
    """
    This function calculates the Quadratic Kappa Metric used for Evaluation

    :param actuals: list of numbers(actual object classes)
                    with values from 0 to N
    :param preds: list of numbers(predicted object classes)
                  with values from 0 to N
    :return: evaluation for prediction with value from
             0.0(absolutely wrong prediction) to
             1.0(absolutely correct prediction)
    """

    w = np.zeros((N, N))

    matrix_o = confusion_matrix(actuals, preds, labels=np.arange(N))

    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

    act_hist = np.zeros(N)
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros(N)
    for item in preds:
        pred_hist[item] += 1

    matrix_e = np.outer(act_hist, pred_hist)

    matrix_e = matrix_e / matrix_e.sum()
    matrix_o = matrix_o / matrix_o.sum()

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * matrix_o[i][j]
            den += w[i][j] * matrix_e[i][j]
    if den == 0:
        return 1
    else:
        return (1 - (num / den)) / 2 + 0.5


def list_of_class_values_from_file(file_path):
    """
    This function extracts a list of group values from csv file with columns
    installation_id and accuracy_group

    :param file_path: path to the file
    :return: list of sorted by installation_id group values
    """

    df = pd.read_csv(file_path)

    df.sort_values(by="installation_id")

    return df["accuracy_group"].values.tolist()


def list_of_class_values_from_df(df):
    """
    This function extracts a list of group values from dataframe with columns
    installation_id and accuracy_group

    :param df: input dataframe
    :return: list of sorted by installation_id group values
    """

    df.sort_values(by="installation_id")

    return df["accuracy_group"].values.tolist()

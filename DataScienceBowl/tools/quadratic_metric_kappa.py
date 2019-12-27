import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix


# This function calculates the Quadratic Kappa Metric used for Evaluation
def quadratic_kappa(input_path, output_path, actuals_name, preds_name, N=4):
    w = np.zeros((N, N))
    actuals = []
    preds = []

    actuals_path = os.path.join(input_path, actuals_name)
    preds_path = os.path.join(output_path, preds_name)

    actuals_df = pd.read_csv(actuals_path)
    preds_df = pd.read_csv(preds_path)

    actuals_list = actuals_df.values.tolist()
    preds_list = preds_df.values.tolist()

    actuals_list.sort()
    preds_list.sort()

    for i in range(len(actuals_list)):
        actuals.append(actuals_list[i][1])
        preds.append(preds_list[i][1])

    matrix_o = confusion_matrix(actuals, preds, labels=np.arange(N))

    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist)

    E = E / E.sum()
    matrix_o = matrix_o / matrix_o.sum()

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * matrix_o[i][j]
            den += w[i][j] * E[i][j]

    return (1 - (num / den)) / 2 + 0.5

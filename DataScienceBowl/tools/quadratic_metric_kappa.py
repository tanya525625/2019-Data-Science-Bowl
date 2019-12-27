import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix

# This function calculates the Quadratic Kappa Metric used for Evaluation
# actuals and preds - lists of numbers(object classes) with values from 0 to N
# return value is in the range 0(absolutely wrong prediction) to 1(absolutely correct prediction)
def quadratic_kappa(actuals, preds, N=4):
    w = np.zeros((N, N))

    O = confusion_matrix(actuals, preds, labels=np.arange(N))

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
    O = O / O.sum()

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * O[i][j]
            den += w[i][j] * E[i][j]
    if den == 0:
        return 1
    else:
        return (1 - (num / den)) / 2 + 0.5

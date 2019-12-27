import numpy as np
import pandas as pd
import os

# This function extracts a list of group values from csv file with columns
# installation_id and accuracy_group
def list_of_class_values(file_path):
    df = pd.read_csv(file_path)

    df.sort_values(by="installation_id")

    return df["accuracy_group"].values.tolist()

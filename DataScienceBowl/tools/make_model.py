import pandas as pd
from sklearn.model_selection import train_test_split

from tools.encoder import make_number_hashes_for_list
from tools.null_processing import find_mean_of_accuracy_group


class ModelMaker:
    """ Class for making models """

    def __init__(self, model_alg, hyperparams, dataset):
        """
        ModelMaker constructor (can take any model_alg
        and corresponding hyperparameters)

        :param model_alg: model algorithm
        :param hyperparams: hyperparams for model
        :param dataset: dataset, which divides to train and test
        """

        self.x_train, self.x_test, self.y_train, self.y_test, \
            self.test_ist_ids = prepare_train_and_test(dataset)
        self.model = model_alg(**hyperparams)
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        """ Method for getting prediction """
        return self.model.predict(self.x_test)


def prepare_train_and_test(dataset):
    """
    Function for splitting dataset
    to train and test datasets

    :param dataset: dataset for splitting
    :return: train and test datasets
    """

    x_train = dataset["installation_id"]
    y_train = dataset["accuracy_group"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.33, random_state=42
    )
    x_test, y_test = remove_duplicate_values_in_test(x_test, y_test)
    x_train = make_number_hashes_for_list(x_train.tolist()).reshape(-1, 1)
    x_test_hash = make_number_hashes_for_list(x_test.tolist()).reshape(-1, 1)

    return x_train, x_test_hash, y_train, y_test, x_test


def remove_duplicate_values_in_test(x, y):
    """
    Function for removing duplicates
    for test dataset

    :param x: index column of test dataset
    :param y: column with values of test dataset
    :return: x_test, y_test
    """

    df = pd.DataFrame(list(y), index=x)
    df.columns = ["accuracy_group"]
    df = find_mean_of_accuracy_group(df)
    df["accuracy_group"] = df.accuracy_group.apply(int)
    return df["installation_id"], df["accuracy_group"]

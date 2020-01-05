import pandas as pd
from sklearn.model_selection import train_test_split

from tools.encoder import make_number_hashes_for_list
from tools.null_processing import find_mean_of_accuracy_group


class ModelMaker:
    """ Class for making models """

    def __init__(self, model_alg, hyperparams, x_train, y_train, x_test):
        """
        ModelMaker constructor (can take any model_alg
        and corresponding hyperparameters)

        :param model_alg: model algorithm
        :param hyperparams: hyperparams for model
        :param dataset: dataset, which divides to train and test
        """

        # self.x_train, self.x_test, self.y_train, self.y_test, \
        #    self.test_ist_ids = prepare_train_and_test(train, test)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

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

    y = dataset['accuracy_group']
    # X = dataset['installation_id']
    X = dataset.drop('accuracy_group', axis=1)
    y.columns = ['accuracy_group']
    x_train, x_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.33, random_state=42)

    # x_test, y_test, x_test_ids = remove_duplicate_values_in_test(x_test, y_test)
    #x_train = make_number_hashes_for_list(x_train.values).reshape(-1, 1)
    #x_test_hash = make_number_hashes_for_list(x_test.values).reshape(-1, 1)

    # return x_train, x_test_hash, y_train, y_test, x_test
    return x_train, x_test, y_train, y_test, x_test


def remove_duplicate_values_in_test(x, y):
    """
    Function for removing duplicates
    for test dataset

    :param x: index column of test dataset
    :param y: column with values of test dataset
    :return: x_test, y_test
    """

    x_test_ids = x["installation_id"].drop_duplicates()
    df = pd.DataFrame(list(y), index=x)
    df.columns = ["accuracy_group"]
    df = find_mean_of_accuracy_group(df)
    df["accuracy_group"] = df.accuracy_group.apply(int)
    return df.index, df["accuracy_group"], x_test_ids


def prepare_hash_train_and_test_kaggle(train_dataset, test_dataset):
    """
    Function for splitting dataset
    to train and test datasets

    :param dataset: dataset for splitting
    :return: train and test datasets
    """
    
    y_train = train_dataset["accuracy_group"]
    x_train = train_dataset.drop("accuracy_group", axis=1)

    # Temporary version
    test_dataset_ids = test_dataset["installation_id"]
    test_dataset_ids.columns = ["installation_id"]

    test_dataset = test_dataset.drop_duplicates()
    #x_train = make_number_hashes_for_list(x_train.values).reshape(-1, 1)
    #x_test_hash = make_number_hashes_for_list(test.values).reshape(-1, 1)

    return x_train, test_dataset, y_train, test_dataset_ids


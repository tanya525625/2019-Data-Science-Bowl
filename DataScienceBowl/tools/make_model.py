from sklearn.model_selection import train_test_split

from tools.encoder import make_number_hashes_for_list


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

        self.x_train, self.x_test, self.y_train, self.y_test = \
            prepare_train_and_test(dataset)
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
    x_train = make_number_hashes_for_list(x_train.tolist()).reshape(-1, 1)
    x_test = make_number_hashes_for_list(x_test.tolist()).reshape(-1, 1)

    return x_train, x_test, y_train, y_test

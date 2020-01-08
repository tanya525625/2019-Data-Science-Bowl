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

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

        self.model = model_alg(**hyperparams)
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        """ Method for getting prediction """
        return self.model.predict(self.x_test)

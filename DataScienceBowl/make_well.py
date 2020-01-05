
import os

from tools.file_worker import read_data
from tools.null_processing import cacheable_drop_nones
from tools.quadratic_metric_kappa import quadratic_kappa
from tools.make_model import ModelMaker,  prepare_train_and_test
from tools.prepare_data import cacheable_prepare_data

from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import json
import gc


def find_well_hyperparams(data_path: str, data_files: list,
                          hyperparams_range: dict):
    """
    Method for finding optimal hyperparameters
    within given hyperparameters range.
    Also can take different model types.
    See get_hyperparams_range method for sample.
        :param data_path: path to data directory
        :param data_files: list of file names of daset elements
        :param hyperparams_range: dict that describes ranges
                                  and sets of params to be tried
        :return: hyperparams that have the best score and their score
    """
    dataset = _prepare_dataset(data_path, data_files)
    hyperparams = _get_min_from_range(hyperparams_range)
    hyperparams_sets = _transform_range_to_set(hyperparams_range)
    hyperparams_range = None
    best_result = _find_best_params(dataset, hyperparams, hyperparams_sets)
    return best_result


def _prepare_dataset(data_path, data_files):
    dataset = cacheable_prepare_data(data_path, data_files)
    x_train, x_test, y_train, y_test, test_ist_ids =\
        prepare_train_and_test(dataset)
    dataset = {'x_train': x_train, 'y_train': y_train,
               'x_test': x_test, 'y_test': y_test}
    return dataset


def _transform_range_to_set(hyperparams_range):
    '''
    Method transforms range (min, max, step) to sets
    (min, min + step, ...)
        :param hyperparams_range: dict that describes ranges
                                and sets of params to be tried
        :return: dict with sets of value of hyperparams
    '''
    ranges = {key: hyperparams_range[key]
              for key in hyperparams_range
              if type(hyperparams_range[key][0]) is int}
    hyperparams_sets = {key: [] for key in ranges}
    for key in ranges:
        for val in range(ranges[key][0], ranges[key][1], ranges[key][2]):
            hyperparams_sets[key].append(val)
    hyperparams_sets.update({key: hyperparams_range[key]
                             for key in hyperparams_range
                             if key not in ranges})
    return hyperparams_sets


def _get_min_from_range(hyperparams_range):
    '''
    Method get initial values of hyperparams from it's ranges.
        :return: dict-hyperparams
    '''
    return {key: hyperparams_range[key][0] for key in hyperparams_range}


def _find_best_params(dataset, hyperparams, hyperparams_sets):
    '''
    Method to try all variants of hyperparams using it's range and choose
    the best params (by quadratic_kappa evaluation)
        :param dataset: dataset for model training and testing
        :param hyperparams: current hyperparams
        :param hyperparams_sets: dict, sets of hyperparams value
        :return: dict with sets of value of hyperparams
    '''
    n = len(hyperparams_sets['model_type'])
    result = {'value': 0, 'hyperparams': None}
    for i in range(n):
        hyperparams['model_type'] = hyperparams_sets['model_type'][i]
        valid_keys = hyperparams['model_type'][1]
        sets = {key: hyperparams_sets[key] for key in valid_keys}
        _recursive_finding(dataset, hyperparams, hyperparams_sets,
                           valid_keys, result, 0)
    return result


def _recursive_finding(dataset, hyperparams, hyperparams_sets,
                       keys, result, index):
    '''
    Method to try all variants of hyperparams using it's range and choose
    the best params (by quadratic_kappa evaluation) in within one model type
        :param dataset: dataset for model training and testing
        :param hyperparams: current hyperparams
        :param hyperparams_sets: dict, sets of hyperparams value
        :param keys: keys of hyperparams_sets dict
        :param result dict to storing the best hyperparams and their score
        :param index: current index
        :return: dict with sets of value of hyperparams
    '''
    if (index >= len(keys)):
        value = _try_params(hyperparams, dataset)
        if (value > result['value']):
            result['value'] = value
            result['hyperparams'] = hyperparams.copy()
            print(value)
        return
    key = keys[index]
    n = len(hyperparams_sets[key])
    for i in range(n):
        hyperparams[key] = hyperparams_sets[key][i]
        _recursive_finding(dataset, hyperparams, hyperparams_sets, keys,
                           result, index + 1)


def _try_params(hyperparams, dataset):
    '''
    Method to make model by given hyperparams and evaluate it by
    quadratic_kappa
        :param hyperparams: current hyperparams
        :param dataset: dataset for model training and testing
        :return: score of hyperparams evaluated by quadratic_kappa
    '''
    params = hyperparams.copy()
    del params['model_type']

    model = ModelMaker(hyperparams['model_type'][0], params,
                       dataset['x_train'],
                       dataset['y_train'],
                       dataset['x_test'])
    prediction = model.predict()
    del model
    del params
    gc.collect()
    return quadratic_kappa(dataset['y_test'], prediction, 4)


def get_hyperparams_range():
    '''
    Method to return sample hyperparams_range
    '''
    valid_keys = ["n_neighbors", "weights", "weights",
                  "algorithm", "leaf_size", "p", "metric"]
    return {
        "model_type": [[KNeighborsClassifier, valid_keys], ],
        "n_neighbors": [1, 10, 1],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [10, 50, 5],
        "p": [2, 3, 1],
        "metric": ["minkowski"],
    }


if __name__ == '__main__':
    input_path = "../Data"
    output_path = os.path.join("..", "Prediction")
    files = ["sample_submission.csv", "test.csv",
             "train.csv", "train_labels.csv"]
    hyperparams_ranges = get_hyperparams_range()
    well_params = find_well_hyperparams(input_path, files, hyperparams_ranges)
    print(well_params['hyperparams'])
    print(well_params['value'])
    del well_params['hyperparams']['model_type'][0]
    str = json.dumps(well_params)
    with open("well_params.json", 'wt') as f:
        f.write(str)
    

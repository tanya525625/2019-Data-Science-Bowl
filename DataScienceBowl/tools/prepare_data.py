from os import path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from tools.file_worker import read_data, FileWorker


def missing_values_table(df):
    """
    The function finds percentage of missing values for each column in df
    """

    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    columns_names = {0: 'Missing Values', 1: '% of Total Values'}
    mis_val_table_ren_columns =\
        mis_val_table.rename(columns=columns_names)

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.
        iloc[:, 1] != 0].\
        sort_values('% of Total Values',
                    ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def prepare_train_data(train, train_labels):
    '''
    Shape DataFrame for prediction function
    '''

    train = train.merge(train_labels[["game_session", 'installation_id',
                                      "accuracy_group"]])
                                      #"event_count",
                                      #"game_time", "type", "world"]])
    train = train.dropna()
    print(train)
    # train = train.drop('game_session', axis=1)

    enc=LabelEncoder()
    for col in list(train):
        if (col == 'installation_id'):
            pass
        else:
            train[col] = enc.fit_transform(train[col])

    inst_id_list = train['installation_id'].tolist()
    for i in range(len(inst_id_list)):
        inst_id_list[i] = int(inst_id_list[i], 16)

    train['installation_id'] = inst_id_list

    return train


def prepare_test_data(test_dataset):
    '''
    Shape DataFrame for prediction function
    '''

    test_dataset = test_dataset.dropna()
    
    enc=LabelEncoder()
    for col in list(test_dataset):
        if (col == 'installation_id'):
            pass
        else:
            test_dataset[col] = enc.fit_transform(test_dataset[col])

    inst_id_list = test_dataset['installation_id'].tolist()
    for i in range(len(inst_id_list)):
        inst_id_list[i] = int(inst_id_list[i], 16)

    test_dataset['installation_id'] = inst_id_list
    
    return test_dataset


def cacheable_prepare_data(data_path, data_files):
    '''
    Function for preparing data to model's work
    with storing results in file 'prepared_data.csv'
    in data_path directory

    :param data_path: path to data directory
    :param data_files: list of files of data
    :return: new train dataset (pd.DataFrame)
    '''
    fw = FileWorker
    cache_path = path.join(data_path, 'prepared_data.csv')
    if path.exists(cache_path):
        train_dataset = fw.read_df(cache_path)
    else:
        data = read_data(data_files, data_path)
        train_dataset = prepare_train_data(data['train.csv'],
                                     data['train_labels.csv'])
        fw.write_df(train_dataset, cache_path)
    return train_dataset


def prepare_data(test_dataset, train_dataset, train_labels, isKaggle):
    train_dataset = merge_train_datasets(train_dataset, train_labels)
    train_dataset.dropna(inplace=True)
    common_dataset = pd.concat([train_dataset, test_dataset], sort=False)

    train_length = len(train_dataset.index)

    enc = LabelEncoder()
    for col in list(common_dataset):
        if col == 'installation_id' or col == "accuracy_group":
            pass
        else:
            common_dataset[col] = enc.fit_transform(common_dataset[col])

    # inst_id_list = common_dataset['installation_id'].tolist()
    # for i in range(len(inst_id_list)):
    #     inst_id_list[i] = int(inst_id_list[i], 16)

    # common_dataset['installation_id'] = inst_id_list

    test_dataset = common_dataset.iloc[train_length:]
    train_dataset = common_dataset.iloc[:train_length]

    if not isKaggle:
        test_dataset = test_dataset.drop('accuracy_group', axis=1)
    test_inst_id_not_enc = test_dataset['installation_id']

    if not isKaggle:
        x_train, x_test, y_train, y_test = prepare_test_for_us(train_dataset)
        test_inst_id_not_enc = x_test
        test_dataset = pd.concat([x_test, y_test], sort=False, axis=1)
        train_dataset = pd.concat([x_train, y_train], sort=False, axis=1)

    test_dataset = encode_inst_id(test_dataset)
    train_dataset = encode_inst_id(train_dataset)
    return test_dataset, train_dataset, test_inst_id_not_enc


def prepare_test_for_us(train_dataset):
    y = train_dataset['accuracy_group']
    X = train_dataset.drop('accuracy_group', axis=1)
    y.columns = ['accuracy_group']
    x_train, x_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test


def merge_train_datasets(train_dataset, train_labels):
    train_dataset = train_dataset.merge(train_labels[["game_session", 'installation_id',
                                      "accuracy_group"]])
    return train_dataset.dropna()


def encode_inst_id(df):
    inst_id_list = df['installation_id'].tolist()
    for i in range(len(inst_id_list)):
        inst_id_list[i] = int(inst_id_list[i], 16)
    df.loc[:, "installation_id"] = inst_id_list

    return df
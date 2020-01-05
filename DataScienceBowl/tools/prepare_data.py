import pandas as pd
from tools.file_worker import read_data, FileWorker
from os import path


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


def prepare_data(train, train_labels):
    '''
    Shape DataFrame for prediction function
    '''

    train = train.merge(train_labels[["game_session", 'installation_id',
                                      "accuracy_group"]])
                                      #"event_count",
                                      #"game_time", "type", "world"]])
    train = train.dropna()
    train = train.drop('event_id', axis=1)

    return train


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
        train_dataset = prepare_data(data['train.csv'],
                                     data['train_labels.csv'])
        fw.write_df(train_dataset, cache_path)
    return train_dataset

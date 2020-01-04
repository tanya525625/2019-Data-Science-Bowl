import pandas as pd
import os


class FileWorker:
    """ Class for working with files """

    @staticmethod
    def read_df(input_path: str, extension="csv"):
        """
        Method for reading dataframes from files

        :param input_path: path to the file
        :param extension: extension of the file, csv by default
        :return: pd.DataFrame, which was read
        """

        return getattr(pd, f"read_{extension}")(input_path)

    @staticmethod
    def write_df(df: pd.DataFrame, output_path: str, extension="csv"):
        """
        Method for writing dataframe

        :param df: dataframe for writing
        :param output_path: path to the file
        :param extension: extension of the file
        """

        if check_format_support(output_path):
            getattr(df, f"to_{extension}")(output_path)


def check_format_support(path):
    """
    Function for checking support of the format by this program

    :param path: path to file for checking
    """

    extension = determine_format(path)
    supported_formats = ("parquet", "json", "csv", "pickle", "msgpack")
    return extension in supported_formats


def determine_format(path: str):
    """
    Function for determining extension of the file in path

    :param path: path to the file
    :return: extension of the file
    """

    path = os.path.splitext(path)
    return path[1][1:]


def read_data(files, input_path):
    """
    Function for reading datasets from files
    All datasets can be received by dereferencing of the dictionary,
    e.g. data_dict['name_of_the_file']

    :param files: list with files names
    :param input_path: path to the files
    :return: dict with dataframes from files
    """

    fw = FileWorker()
    data_dict = {}
    for file in files:
        df = fw.read_df(os.path.join(input_path, file))
        data_dict.update({file: df})
    return data_dict


def write_submission(inst_ids: list, prediction: list, path_to_file: str):
    df = pd.DataFrame(prediction, index=inst_ids)
    fw = FileWorker()
    fw.write_df(df, path_to_file)

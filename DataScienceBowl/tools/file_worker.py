import pandas as pd
import os


class FileWorker:
    @staticmethod
    def read_df(input_path, extension="csv"):
        return getattr(pd, f"read_{extension}")(input_path)

    @staticmethod
    def write_df(df, extension, output_path):
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


def determine_format(path):
    path = str(path)
    path = os.path.splitext(path)
    extension = path[1][1:]
    return


def read_data(files, input_path):
    fw = FileWorker()
    data_dict = {}
    for file in files:
        df = fw.read_df(os.path.join(input_path, file))
        data_dict.update({file: df})
    print(data_dict)
    return data_dict



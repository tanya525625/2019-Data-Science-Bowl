def drop_nones(data: dict):
    """
    Function for dropping nones (information about games,
    which were not launched)

    :param data: dataframes with data
    :return: new train dataset (pd.DataFrame)
    """

    train_df = data["train.csv"]
    train_labels_df = data["train_labels.csv"]

    train_df = if_game_was_launched(train_df)
    train_aggs = find_mean_of_not_launched_games(train_df)
    # train_labels_df = find_mean_of_accuracy_group(train_labels_df)
    train_aggs = merge_dataframes(train_aggs, train_labels_df)
    train_aggs.dropna(inplace=True)

    return train_aggs


def if_game_was_launched(df):
    """
    Function for determining if the game was launched

    :param df: dataframe for investigation
    :return: dataframe with the information of lauched games
    """

    df["not_launched"] = True
    # if in the dataset there is information about launching
    df.loc[
        df["event_data"].str.contains("false") &
        df["event_code"].isin([4100, 4110]),
        "not_launched",
    ] = False

    return df


def find_mean_of_not_launched_games(df):
    """
    Function for finding mean value of not_launched games by installation_id

    :param df: dataframe for ivestigation
    :return: new dataframe
    """

    df = df.groupby("installation_id").agg({"not_launched": "mean"})
    df = df.reset_index()
    return df


def find_mean_of_accuracy_group(df):
    """
    Function for finding mean value of accuracy_group by installation_id

    :param df: dataframe for investigation
    :return:
    """

    df = df.groupby("installation_id").\
        agg({"accuracy_group": "mean"}).reset_index()
    return df


def merge_dataframes(df1, df2):
    """
    Function for merging (left join) two dataframes

    :param df1: first dataframe
    :param df2: second dataframe
    :return: merged dataframes
    """

    return df1.merge(df2[["installation_id", "accuracy_group"]], how="left")

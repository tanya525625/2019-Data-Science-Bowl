def drop_nones(data: dict):
    """
    Function for dropping nones (information about games,
    which were not launched)

    :param data: dataframes with data
    :return: new train dataset (pd.DataFrame)
    """

    train_df = data["train.csv"]
    train_labels_df = data["train_labels.csv"]

    train_df["not_launched"] = True
    # if in train dataset there is information about launching
    train_df.loc[
        train_df["event_data"].str.contains("false")
        & train_df["event_code"].isin([4100, 4110]),
        "not_launched",
    ] = False

    # finding mean value of not_launched games by installation_id
    train_aggs = (
        train_df.groupby("installation_id").agg({"not_launched": "mean"}).reset_index()
    )

    # finding mean value of accuracy_group by installation_id
    train_labels_df = (
        train_labels_df.groupby("installation_id")
        .agg({"accuracy_group": "mean"})
        .reset_index()
    )

    train_aggs = train_aggs.merge(
        train_labels_df[["installation_id", "accuracy_group"]], how="left"
    )
    train_aggs.dropna(inplace=True)

    return train_aggs

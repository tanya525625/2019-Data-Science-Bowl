import pandas as pd
import numpy as np

from collections import Counter

from sklearn.model_selection import train_test_split


def prepare_train_dataset_due_to_train_labels(train, train_labels):
    """
    Function for preparation train dataset,
    filters its values due to the fact of there existence
    in train_labels dataset

    :param train: train dataset, pd.DataFrame
    :param train_labels: train_labels dataset, pd.DataFrame
    :return: filtered train dataset
    """

    train = train[
        train["installation_id"].isin(list(train_labels["installation_id"].unique()))
    ]

    return train


def prepare_train_dataset_and_test(train, test):
    """
    Function for train and test preparation
    for model working

    :param train: train dataset, pd.DataFrame
    :param test: test dataset, pd.DataFrame
    :return: train and test datasets
    """

    # dropping data that didn't have an assesment
    keep_id = train[train.type == "Assessment"][["installation_id"]].drop_duplicates()
    train = pd.merge(train, keep_id, on="installation_id", how="inner")

    # convert timestamp to datetime
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])

    return train, test


def find_unique_values(train, test):
    """
    Function for finding unique values in train
    and test datasets for further hashing

    :param train: train dataset, pd.DataFrame
    :param test: test dataset, pd.DataFrame
    :return: lists with unique values in train and test
    """

    # make a list with all the unique 'titles' from the train and test
    activities_in_train = set(train["title"].value_counts().index)
    activities_in_test = set(test["title"].value_counts().index)

    all_activities = list(activities_in_train.union(activities_in_test))

    # make a list with all the unique 'event_code' from the train and test
    event_codes_in_train = set(train["event_code"].value_counts().index)
    event_codes_in_test = set(test["event_code"].value_counts().index)

    all_event_codes = list(event_codes_in_train.union(event_codes_in_test))

    return all_activities, all_event_codes


def process_data(train, test):
    """
    Function for processing train and test datasets

    :param train: train dataset, pd.DataFrame
    :param test: test dataset, pd.DataFrame
    :return: datasets for model working
    """

    # lists of unique values from 'title' and 'event_code'
    all_activities, all_event_codes = find_unique_values(train, test)

    # encode 'title' values
    activities_values, activities_names = make_hashes(all_activities)

    # encode 'title' in train and tets datasets
    train, test = encode_data(train, test, activities_values)

    # generate dictionary with win codes for each 'title'
    win_codes = make_win_codes(activities_values)

    # for each installation_id process data about its sessions
    new_train = []

    for i, (ins_id, user_sample) in enumerate(
        train.groupby("installation_id", sort=False)
    ):
        # user_sample is a DataFrame that consists of only one installation_id's game sessions
        new_train += process_assessments(
            user_sample, all_activities, all_event_codes, activities_names, win_codes
        )

    new_train = pd.DataFrame(new_train)

    # the same for test dataset
    new_test = []

    for ins_id, user_sample in test.groupby("installation_id", sort=False):
        new_test.append(
            process_assessments(
                user_sample,
                all_activities,
                all_event_codes,
                activities_names,
                win_codes,
                test_set=True,
            )
        )

    new_test = pd.DataFrame(new_test)

    # create a list of the features
    features = list(new_train.columns.values)
    features.remove("accuracy_group")

    # removes accuracy_group from the train data
    X_train = new_train[features]
    # create a variable to contain just the accuracy_group label of the train data
    y_train = new_train["accuracy_group"]
    # remove accuracy_group from the test data
    X_test = new_test[features]
    y_test = new_test["accuracy_group"]

    return X_train, y_train, X_test, y_test


def make_hashes(all_activities):
    """
    Function for making hashes for data encoding

    :param all_activities: data for encoding
    :return: dicts with hashes for data
    """

    # create a dictionary numerating the titles
    activities_codes = dict(zip(all_activities, np.arange(len(all_activities))))
    activities_names = dict(zip(np.arange(len(all_activities)), all_activities))

    return activities_codes, activities_names


def encode_data(train, test, activities_values):
    """
     Function for data encoding

    :param train: train dataset, pd.DataFrame
    :param test: test dataset, pd.DataFrame
    :param activities_values: hashes
    :return: encoded train and test datasets
    """

    # replace the text titles withing the number titles from the dict
    train["title"] = train["title"].map(activities_values)
    test["title"] = test["title"].map(activities_values)

    return train, test


def make_win_codes(activities_hashes):
    """
    Function for making win_codes for each value
    from activities_hashes

    :param activities_hashes: hashes
    :return: win_codes
    """

    # makes a dict with values where each element equals to 4100
    win_codes = dict(
        zip(
            activities_hashes.values(),
            (4100 * np.ones(len(activities_hashes))).astype("int"),
        )
    )

    # the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_codes[activities_hashes["Bird Measurer (Assessment)"]] = 4110

    return win_codes


def process_assessments(
    user_sample,
    all_activities,
    all_event_codes,
    activities_names,
    win_codes,
    test_set=False,
):
    """
    Function for proccessing assessments,
    makes features

    :param user_sample: pd.DataFrame from train or test where
                        data is for only one installation_id
    :param all_activities: unique activities from all datasets
    :param all_event_codes: unique event codes from all datasets
    :param activities_names: activities names with hashes
    :param win_codes: winner codes
    :param test_set: True if test is required
    :return: processed assesments
    """

    last_activity = 0
    user_activities_count = {"Clip": 0, "Activity": 0, "Assessment": 0, "Game": 0}

    # calculation of time which was spent in each activity
    time_spent_each_act = {actv: 0 for actv in all_activities}
    event_code_count = {eve: 0 for eve in all_event_codes}

    # initialisation
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_incorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    durations = []

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby("game_session", sort=False):
        # i is a game_session id
        # session is a DataFrame that contains data about only one game_session

        # type of current session
        session_type = session["type"].iloc[0]
        # type of activity of current session
        session_title = session["title"].iloc[0]

        # current session time in seconds
        if session_type != "Assessment":
            # how much time spent on current session
            time_spent = int(session["game_time"].iloc[-1] / 1000)
            # how much time spent on each activity
            time_spent_each_act[activities_names[session_title]] += time_spent

        # for each assessment the features are processing
        if (session_type == "Assessment") & (test_set or len(session) > 1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f"event_code == {win_codes[session_title]}")

            # count the number of wins and the number of losses
            true_attempts = all_attempts["event_data"].str.contains("true").sum()
            false_attempts = all_attempts["event_data"].str.contains("false").sum()

            # copy a dict to use as feature template
            features = user_activities_count.copy()
            features.update(time_spent_each_act.copy())
            features.update(event_code_count.copy())

            # add title as feature, title represents the name of the game
            features["session_title"] = session["title"].iloc[0]

            # Add new features
            # information about correct trials
            features["accumulated_correct_attempts"] = accumulated_correct_attempts
            # information about incorrect trials
            features["accumulated_incorrect_attempts"] = accumulated_incorrect_attempts

            accumulated_correct_attempts += true_attempts
            accumulated_incorrect_attempts += false_attempts

            # the time spent in the app so far
            if durations == []:
                features["duration_mean"] = 0
            else:
                features["duration_mean"] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

            # the accuracy is all the time wins divided by the all time attempts
            features["accumulated_accuracy"] = (
                accumulated_accuracy / counter if counter > 0 else 0
            )
            accuracy = (
                true_attempts / (true_attempts + false_attempts)
                if (true_attempts + false_attempts) != 0
                else 0
            )
            accumulated_accuracy += accuracy

            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features["accuracy_group"] = 0
            elif accuracy == 1:
                features["accuracy_group"] = 3
            elif accuracy == 0.5:
                features["accuracy_group"] = 2
            else:
                features["accuracy_group"] = 1
            features.update(accuracy_groups)
            accuracy_groups[features["accuracy_group"]] += 1

            # mean of the all accuracy groups of this player
            features["accumulated_accuracy_group"] = (
                accumulated_accuracy_group / counter if counter > 0 else 0
            )
            accumulated_accuracy_group += features["accuracy_group"]

            # how many actions the player has done so far
            features["accumulated_actions"] = accumulated_actions

            # if it's a test set, all sessions belong to the final dataset
            if test_set:
                all_assessments.append(features)
            # it it's a train, needs to be passed must exist an event_code 4100 or 4110
            elif true_attempts + false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # how many actions was made in each event_code
        n_of_event_codes = Counter(session["event_code"])

        for key in n_of_event_codes.keys():
            event_code_count[key] += n_of_event_codes[key]

        # how many actions the player has done
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1

    # if it's the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]

    return all_assessments


def split_train_and_test(dataset):
    """
    Function for splitting dataset
    to train and test datasets

    :param dataset: dataset for splitting
    :return: train and test datasets
    """

    y = dataset["accuracy_group"]
    # dropping 'accuracy_group' to make X_train
    X = dataset.drop("accuracy_group", axis=1)
    y.columns = ["accuracy_group"]

    # splitting dataset for training
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    return x_train, x_test, y_train, y_test

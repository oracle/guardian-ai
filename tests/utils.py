#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import datetime
import numbers
import random

import numpy as np
import pandas as pd
import pytz


def map_col_types(col_types):
    # Cast columns to proper pandas dtypes
    dtypes = {
        "int": "int64",
        "str": "object",
        "float": "float64",
        "bool": "bool",
        "datetime": "datetime",
        "date": "date",
        "time": "time",
        "datetimez": "datetimez",
        "Timestamp": "Timestamp",
        "timedelta": "timedelta",
    }
    return [dtypes[col_type] for col_type in col_types]


def generate_null(datetime_col, null_ratio):
    num_sample = datetime_col.shape[0]
    num_nulls = int(num_sample * null_ratio)
    num_nans = random.randint(0, num_nulls)
    num_nats = num_nulls - num_nans

    # Getting ranodm indices
    nans_ind = random.sample(range(0, num_sample), num_nans)
    nats_ind = random.sample(range(0, num_sample), num_nats)

    # Assigning nans and nats
    # NOTE: we are using .loc for avoiding dataframe related warnings
    datetime_col.loc[nans_ind] = np.nan
    datetime_col.loc[nats_ind] = pd.NaT

    return datetime_col


def get_dummy_dataset(
    n_samples=5000,
    n_features=10,
    n_classes=2,
    types=[str, float, bool, int],
    content=[],
    contain_null=False,
    null_ratio=0.3,
    dtime_types=[],
    tz_aware=False,
    reg_range=10.0,
    cat_range=30,
    random_seed=9999,
    imb_factor=1.0,
    task="classification",
    **kwargs,
):
    """
    Generates a dummy dataset and returns its corresponding ope/oml
    dataframe:
    dataset shape n_samples x n_features.

    types: column types you wish to generate (random number of columns=
    n_features types are generated, with at least one of each type).

    content: list of tuples (dtype, feature) specifying bad column
    features. Features can be 'const' - to make all values in column
    constant, or value between 0 and 1 which indicates percentage of
    missing values in a column

    dtime_types: datetime column types to generate. Acceptable types
    are: ['datetime', 'date', 'time', 'timedelta', 'datetimetz']

    n_classes: number of target classes (only used for classification)

    reg_range: range of target for regression datasets, not used for
               classification

    cat_range: maximum number of unique values for the categorical
               features

    imb_factor: ~ class_ratio = minority_class_size/majority_class_size
    approximately controls dataset target imbalance
    (only used for classification).

    """
    np.random.seed(random_seed)
    allowed_dtime_types = [
        "datetime",
        "date",
        "time",
        "timedelta",
        "datetimez",
        "Timestamp",
    ]

    # sanity checks
    assert n_samples >= n_classes, "Number of samples has to be greater than num of classes"
    assert (imb_factor > 0) and (imb_factor <= 1.0), "imb_factor has to be in range of (0, 1.0]"
    assert len(types) == len(set(types)), "types inside the list must be unique"
    assert len(dtime_types) == len(set(dtime_types)), "dtime_types inside the list must be unique"
    assert (
        len(dtime_types) + len(types) <= n_features
    ), "provided number of feature types is more than n_features"
    assert task in [
        "classification",
        "regression",
        "anomaly_detection",
    ], "Task must be one of classification or regression"
    assert all(
        x for x in dtime_types if x in allowed_dtime_types
    ), "dtime_types: {} outside of allowed: {}".format(dtime_types, allowed_dtime_types)

    extra_types, extra_feats, extra_cols = [], [], 0
    if content != []:
        extra_cols = len(content)
        extra_types = [x for x, _ in content]
        extra_feats = [x for _, x in content]

    # target labels for the dataset
    if task == "classification" or task == "anomaly_detection":
        # assign class counts based on geometric distribution of classes based on imb_factor
        class_weights = np.geomspace(imb_factor, 1.0, num=n_classes)
        class_counts = [max(1, int(n_samples * x / np.sum(class_weights))) for x in class_weights]
        class_excess = np.sum(class_counts) - n_samples
        class_counts[-1] -= class_excess

        # create labels based on class counts and shuffle them
        y = np.hstack([np.full((1, count), cl) for cl, count in enumerate(class_counts)]).ravel()
        np.random.shuffle(y.astype(int))
        y = y.tolist()
    elif task == "regression":
        # noise between (-reg_range/2, reg_range/2) for regression
        y = reg_range * np.random.random(size=(1, n_samples, 1)) + reg_range / 2.0
        y = y.reshape(1, n_samples).ravel().tolist()

    # tally total number of features
    all_feat_types = types + dtime_types + extra_types
    total_feat_types = len(types) + len(dtime_types)
    if total_feat_types > 0:
        feat_col_types = np.random.choice(
            range(0, total_feat_types), size=n_features - total_feat_types
        ).tolist()
        feat_col_types += list(range(0, total_feat_types))  # to ensure at least one of each type

    else:
        feat_col_types = []
    feat_col_types += list(range(total_feat_types, total_feat_types + len(extra_types)))
    features = []
    col_types = []
    tz = {}
    # extra_features provided in content, and certain datetime columns are handled differently
    # they get added as pandas Series or DataFrames to rest of features in the end
    special_cols_num, special_pd_df = [], []
    extra_features = pd.DataFrame()
    for i, t in enumerate(feat_col_types):
        assert t < total_feat_types + len(extra_types)
        typ = all_feat_types[t]
        if typ is str:
            high_val = np.random.randint(3, cat_range)
            feat = np.random.randint(0, high_val, size=n_samples).tolist()
            feat = ["STR{}".format(val) for val in feat]
        elif typ is int:
            low_val = np.random.randint(-50000, -10)
            high_val = np.random.randint(10, 50000)
            feat = np.random.randint(low_val, high_val, size=n_samples).tolist()
        elif typ is float:
            feat = np.random.rand(n_samples).tolist()
        elif typ is bool:
            feat = np.random.randint(0, 2, size=n_samples).tolist()
            feat = [bool(val) for val in feat]
        elif typ in allowed_dtime_types:
            if typ == "datetime":
                # generating random datetime
                deltas = random.sample(range(1, 172800000), n_samples)
                d1 = datetime.datetime.now() - datetime.timedelta(days=2000)
                d2 = datetime.datetime.now()
                generated_datetime = []
                for d in deltas:
                    generated_datetime.append(d1 + datetime.timedelta(seconds=d))
                feat = generated_datetime
            elif typ == "timedelta":
                feat = n_samples * [datetime.timedelta()]
            elif typ == "time":
                feat = n_samples * [datetime.time()]
            elif typ == "date":
                feat = n_samples * [datetime.date(2019, 9, 11)]
            elif typ == "datetimez":
                special_cols_num.append(i)
                special_pd_df.append(pd.date_range(start=0, periods=n_samples, tz="UTC"))
                feat = n_samples * [
                    datetime.date(2019, 9, 11)
                ]  # needs to be handled in special way b/c it's already pandas obj
            else:
                raise Exception("Unrecognized datetime type of column")
        else:
            raise Exception("Unrecognized type of column")

        # If index reached the last extra_col number of feature types, start modifying features
        # and adding them to extra_features DataFrame instead of list of features
        if extra_cols > 0 and i >= (len(feat_col_types) - extra_cols):
            feat_idx = i - (len(feat_col_types) - extra_cols)
            if isinstance(extra_feats[feat_idx], numbers.Number):
                # missing values given by extra_feats[feat_idx] percentage of instances
                assert (
                    extra_feats[feat_idx] <= 1.0 and extra_feats[feat_idx] >= 0
                ), "feature in content has to be ratio between 0 and 1"
                ids = np.random.choice(
                    range(0, n_samples), size=int(extra_feats[feat_idx] * n_samples)
                ).astype(int)
                dtype = map_col_types([extra_types[feat_idx].__name__])[0]
                feat = pd.Series(data=np.array(feat), dtype=dtype)
                feat[ids] = np.nan
            elif extra_feats[feat_idx] == "const":
                # constant column, set all rows to be same as the first instance
                dtype = map_col_types([extra_types[feat_idx].__name__])[0]
                feat = pd.Series(data=np.array(feat), dtype=dtype)
                feat = feat[0]
            extra_features[i] = feat
        else:  # add features to the list
            features.append(feat)
            col_types.append(type(feat[0]).__name__)

    # if task == 'regression':
    #     # Add scaled target column for regression so that score is positive
    #     features.append([-0.5*x for x in y])
    #     col_types.append('float') # target column type is int

    # Add target column and convert all types to pandas dtypes
    features.append(y)
    col_types.append("int" if task == "classification" else "float")  # target column type is int
    pd_col_types = map_col_types(col_types)
    pd_df = pd.DataFrame(features).T  # transpose to get samples x features
    num_feats = len(features) - 1
    columns = list(range(0, num_feats)) if num_feats > 0 else []
    columns = columns + ["target"]
    pd_df.columns = columns  # rename columns

    # handle special column from datettime: replace placeholder with pandas.date_range columns
    for i, col in enumerate(special_cols_num):
        pd_df[col] = special_pd_df[i]
        pd_col_types[col] = pd_df.dtypes[col]

    # assign datatypes to pd dataframe for non-datetime types
    columns_types_all = list(zip(columns, pd_col_types))
    columns_types_nodtime = [
        (name, typ) for (name, typ) in columns_types_all if typ not in allowed_dtime_types
    ]
    columns_types_dtime = [
        (name, typ) for (name, typ) in columns_types_all if typ in allowed_dtime_types
    ]
    pd_df = pd_df.astype(dict(columns_types_nodtime))  # cast types on non-dtime columns

    # assign datatypes to pd dataframe only for datetime types
    for col, col_type in columns_types_dtime:
        if col_type == "timedelta":
            pd_df[col] = pd.to_timedelta(pd_df[col], errors="coerce")
        elif col_type == "datetimez":
            pd_df[col] = pd_df[col]
        elif col_type == "datetime":
            pd_df[col] = pd.to_datetime(pd_df[col], errors="coerce")
            if contain_null:
                pd_df[col] = generate_null(pd_df[col], null_ratio)
            if tz_aware:
                tz[str(col)] = pytz.all_timezones[np.random.randint(len(pytz.all_timezones))]
        else:
            pd_df[col] = pd.to_timedelta(pd_df[col], errors="coerce")

    # add extra features columns that were provided by content
    pd_df[pd_df.shape[1] + extra_features.columns] = extra_features

    # Convert all the column names to string type (mainly for FS min_features [] tests)
    pd_df.columns = [str(col) for col in pd_df.columns]

    if tz_aware:
        return pd_df.drop(["target"], axis=1), pd_df["target"], tz
    else:
        return pd_df.drop(["target"], axis=1), pd_df["target"]

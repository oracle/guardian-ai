#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
from enum import Enum
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


class DataSplit(Enum):
    """
    Prepare data splits. The main idea here is that we need to carve out a subset of the
    target model's training data for training and testing the attack (attack_train_in and
    attack_test_in). The part of the target model's training data that is not used for the
    attacks is target_additional_train. We also need to set aside some data that was not used
    for training the target model (attack_train_out and attack_test_out). Finally, we need data
    for tuning and testing the target model itself (target_valid, target_test).
    Note that we first do these finer granularity splits, and then merge them to form the
    appropriate train and test sets for the target model and the attack model.

    This is a convenience class for specifying the data split ratios. This works for the attacks
    implemented currently, but we can change or use another split for future attacks.
    This is why the Dataset class implements more general data splitting and merging functions.

    """

    ATTACK_TRAIN_IN = 0
    ATTACK_TRAIN_OUT = 1
    ATTACK_TEST_IN = 2
    ATTACK_TEST_OUT = 3
    TARGET_ADDITIONAL_TRAIN = 4
    TARGET_VALID = 5
    TARGET_TEST = 6


class TargetModelData:
    """
    Convenience class to easily pass around the dataset prepared for training and testing
    the target model
    """

    def __init__(
        self,
        X_target_train,
        y_target_train,
        X_target_valid,
        y_target_valid,
        X_target_test,
        y_target_test,
    ):
        """
        Create Target Model Data
        All X variables are {array-like, sparse matrix} of shape (n_samples, n_features),
        where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Parameters
        ----------
        X_target_train: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input variables used to train the target model.
        y_target_train: ndarray of shape (n_samples,)
            Output labels used to train the target model.
        X_target_valid: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input variables used to tune the target model.
        y_target_valid: ndarray of shape (n_samples,)
            Output variables used to tune the target model.
        X_target_test: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input variables used to test the target model.
        y_target_test: ndarray of shape (n_samples,)
            Output variables used to test the target model.

        """
        self.X_target_train = X_target_train
        self.y_target_train = y_target_train
        self.X_target_valid = X_target_valid
        self.y_target_valid = y_target_valid
        self.X_target_test = X_target_test
        self.y_target_test = y_target_test


class AttackModelData:
    """
    Convenience class to easily pass around the dataset prepared for training and testing
    the attack model
    """

    def __init__(
        self,
        X_attack_train,
        y_attack_train,
        y_membership_train,
        X_attack_test,
        y_attack_test,
        y_membership_test,
    ):
        """
        Create Attack Model Data

        All X variables are {array-like, sparse matrix} of shape (n_samples, n_features),
        where `n_samples` is the number of samples and n_features` is the number of features.
        All y variables are ndarray of shape (n_samples,)

        Parameters
        ----------
        X_attack_train: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input variables for the dataset on which we want to train
            the attack model. These are the original features (not attack/membership features)
        y_attack_train: ndarray of shape (n_samples,)
            Output labels for the dataset on which we want to train
            the attack model. These are the original labels (not membership labels)
        y_membership_train: ndarray of shape (n_samples,)
            Membership labels for the dataset on which we want to train
            the attack model. These are binary and indicate whether the data point was included
        X_attack_test: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input variables for the dataset on which to run the attack model.
            These are the original features (not attack/membership features)
        y_attack_test: ndarray of shape (n_samples,)
            Output labels for the dataset on which to run the attack model.
            These are the original labels (not membership labels)
        y_membership_test: ndarray of shape (n_samples,)
            Membership labels for the dataset on which we want to run
            the attack model. These are binary and indicate whether the data point was included
            in the training dataset of the target model, and helps us evaluate the attack model's
            accuracy.

        """
        self.X_attack_train = X_attack_train
        self.y_attack_train = y_attack_train
        self.y_membership_train = y_membership_train
        self.X_attack_test = X_attack_test
        self.y_attack_test = y_attack_test
        self.y_membership_test = y_membership_test


class Dataset:
    """
    Wrapper for the dataset that also maintains various data splits that are required for
    carrying out the attacks.
    Also implements utility methods for generating attack sets
    """

    def __init__(self, name: str = None, df_x=None, df_y=None):
        """
        Create the dataset wrapper.

        Parameters
        ----------
        name: str
            Name for this dataset.
        df_x: {array-like, sparse matrix} of shape (n_samples, n_feature),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        df_y: darray of shape (n_samples,)
            Output labels.

        """
        self.name = name
        self.df_x, self.df_y = df_x, df_y
        self.splits = {}

    def split_dataset(
        self, seed: int, split_array: List[float], split_names: List[str] = None
    ):
        """
        Splits dataset according to the specified fractions.

        Parameters
        ----------
        seed: int
            Random seed for creating the splits.
        split_array: List[float]
            Array of fractions to split the data in. Must sum to 1.
        split_names: List[str]
            Names assigned to the splits.

        Returns
        -------
        dict
            dict of string to tuple of df_x and df_y of the splits
            Dictionary of splits, with keys as the split names and values as the splits

        """
        assert np.round(np.sum(split_array), 3) == 1.0
        if split_names is not None:
            assert len(split_array) == len(split_names)

        x_2, y_2 = (
            self.df_x,
            self.df_y,
        )  # using these variables as portion to be split next
        test_size = np.sum(split_array[1:])
        for i in range(len(split_array)):
            split_name = split_names[i] if split_names is not None else "d" + str(i)
            if test_size != 0:
                x_1, x_2, y_1, y_2 = train_test_split(
                    x_2, y_2, test_size=test_size, random_state=seed
                )
                self.splits[split_name] = [x_1, y_1]
                test_size = 1 - (
                    split_array[i + 1] / np.sum(split_array[i + 1 :])
                )  # calculate the new ratio, based on the size of the remaining data
            else:
                self.splits[split_name] = [x_2, y_2]  # add the last split
        for key in self.splits.keys():
            print(key + "\t" + str(len((self.splits[key])[1])))
        return self.splits

    def _sample_from_split(self, split_name: str, frac: float = None, seed: int = 42):
        """
        Sample a small fraction of the data.

        Parameters
        ----------
        split_name: str
            The dataset from which we're selecting the sample.
        frac: float
            fraction to sample.
        seed: int
            random seed.

        Returns
        -------
        {array-like, sparse matrix} of shape (n_samples, n_feature), darray of shape (n_samples,)
            data sample

        """
        x_split, y_split = self.splits[split_name]
        x_sample, _, y_sample, _ = train_test_split(
            x_split, y_split, test_size=1 - frac, random_state=seed
        )
        return x_sample, y_sample

    def get_merged_sets(self, split_names: List[str]):
        """
        Merge multiple splits of data.

        Parameters
        ----------
        split_names: List[str]
            Names of splits to be merged.

        Returns
        -------
        {array-like, sparse matrix} of shape (n_samples, n_feature), darray of shape (n_samples,)
            Merged datasets.

        """
        x_splits = []
        y_splits = []
        for split_name in split_names:
            x_split, y_split = self.splits[split_name]
            x_splits.append(x_split)
            y_splits.append(y_split)
        x_merged = (
            sp.vstack(x_splits)
            if (sp.issparse(x_splits[0]))
            else np.concatenate(x_splits)
        )
        y_merged = pd.concat(y_splits)
        return x_merged, y_merged

    def _create_attack_set(self, X_attack_in, y_attack_in, X_attack_out, y_attack_out):
        """
        Given the splits that correspond to attack in and out sets, generate the full attack
        set.

        Parameters
        ----------
        X_attack_in: {array-like, sparse matrix} of shape (n_samples, n_feature),
            Input features of the attack data points included during training
            the target model.
        y_attack_in: darray of shape (n_samples,)
            Output labels of the attack data points included during training
            the target model.
        X_attack_out: {array-like, sparse matrix} of shape (n_samples, n_feature),
            Input features of the attack data points not included
            during training the target model.
        y_attack_out: darray of shape (n_samples,)
            Output labels of the attack data points not included
            during training the target model.

        Returns
        -------
        {array-like, sparse matrix} of shape (n_samples, n_feature), darray of shape (n_samples,), darray of shape (n_samples,)

        """
        y_membership = []
        X_attack = (
            sp.vstack((X_attack_in, X_attack_out))
            if (sp.issparse(X_attack_out))
            else np.concatenate((X_attack_in, X_attack_out))
        )
        y_attack = pd.concat([y_attack_in, y_attack_out])

        for _i in range((X_attack_in).shape[0]):
            y_membership.append(1)
        for _i in range((X_attack_out).shape[0]):
            y_membership.append(0)

        return X_attack, y_attack, y_membership

    def create_attack_set_from_splits(self, attack_in_set_name, attack_out_set_name):
        """
        Given the splits that correspond to attack in and out sets, generate the full attack
        set.

        Parameters
        ----------
        attack_in_set_name:
            Dataset that was included as part of the training set of the target model.
        attack_out_set_name:
            Dataset that was not included as part of the training set of the target model.

        Returns
        -------
        {array-like, sparse matrix} of shape (n_samples, n_feature), darray of shape (n_samples,), darray of shape (n_samples,)
            Input features and output labels of the attack data points, along with their
            membership label (0-1 label that says whether or not they were included during training
            the target model)

        """
        X_attack_in, y_attack_in = self.splits[attack_in_set_name]
        X_attack_out, y_attack_out = self.splits[attack_out_set_name]
        return self._create_attack_set(
            X_attack_in, y_attack_in, X_attack_out, y_attack_out
        )

    # sample a fraction of the train set to create attack in set, and merge with the given
    # out set to generate the full attack set
    def create_attack_set_by_sampling_from_train(
        self, train_set_name, train_fraction, attack_out_set_name, seed=42
    ):
        X_attack_in, y_attack_in = self._sample_from_split(
            train_set_name, frac=train_fraction, seed=seed
        )
        X_attack_out, y_attack_out = self.splits[attack_out_set_name]
        return self._create_attack_set(
            X_attack_in, y_attack_in, X_attack_out, y_attack_out
        )

    @abstractmethod
    def load_data(
        self,
        source_file,
        header: bool = None,
        target_ix: int = None,
        ignore_ix: List[int] = None,
    ):
        """
        Method that specifies how the data should be loaded. Mainly applicable for tabular data

        Parameters
        ----------
        source_file: os.path
            Filename of the source file.
        header: bool
            Whether to contain header.
        target_ix: int
            Index of the target variable.
        ignore_ix: List[int]
            Indices to be ignored.

        Returns
        -------
        pandas dataframe of shape (n_samples, n_feature), pandas df of shape (n_samples,)
            Input features and output labels.

        """
        pass


class ClassificationDataset(Dataset):
    """
    Generic classification dataset in a tabular format, read in a somewhat consistent manner
    """

    def __init__(self, name, df_x=None, df_y=None):
        """
        Create a Classification Dataset wrapper.

        Parameters
        ----------
        name: str
            Name of the dataset
        df_x: {array-like, sparse matrix} of shape (n_samples, n_feature),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        df_y: darray of shape (n_samples,)
            Output labels.

        """
        self.df_x = df_x
        self.df_y = df_y
        self.column_transformer = None
        self.label_encoder = None
        self.target_model_data = None
        self.attack_model_data = None
        super(ClassificationDataset, self).__init__(name)

    def load_data_from_df(self, input_features, target):
        """
        Load data from another data frame.

        Parameters
        ----------
        input_features: pandas.DataFrame
        target: pandas.DataFrame

        Returns
        -------
        None

        """
        self.df_x = input_features
        self.df_y = target

    def load_data(
        self,
        source_file,
        contains_header: bool = False,
        target_ix: int = None,
        ignore_ix: List[int] = None,
    ):
        """
        Method that specifies how the data should be loaded. Mainly applicable for tabular data.

        Parameters
        ----------
        source_file: os.path
            Filename of the source file.
        contains_header: bool
            Whether to contain header.
        target_ix: int
            Index of the target variable.
        ignore_ix: List[int]
            Indices to be ignored.

        Returns
        -------
        pandas dataframe of shape (n_samples, n_feature), pandas df of shape (n_samples,)
            Input features and output labels.

        """
        df = None
        if source_file.endswith(".csv"):
            if contains_header:
                df = pd.read_csv(
                    source_file, sep=",", skiprows=1, header=None, encoding="utf-8"
                )  # ignore the headers, especially when reading lots of datasets.
            else:
                df = pd.read_csv(source_file, sep=",", header=None, encoding="utf-8")
        elif source_file.endswith(".arff"):
            data = arff.loadarff(source_file)
            df = pd.DataFrame(data[0])
        else:
            raise ValueError

        # first, find the y index and remove it to get x
        y_ix = target_ix if target_ix is not None else len(df.columns) - 1
        self.df_y = df.iloc[:, y_ix]
        if isinstance(self.df_y[0], bytes):
            self.df_y = self.df_y.str.decode("utf-8")
        self.df_x = df.drop(df.columns[y_ix], axis=1)

        # next remove the ones that need to be ignored.
        if ignore_ix is not None:
            self.df_x = self.df_x.drop(ignore_ix, axis=1)

    def get_column_transformer(self):
        """
        Transforming categorical and numerical features.

        Returns
        -------
        Pipeline
            pipeline of column transformers.

        """
        if self.column_transformer is None:
            assert self.df_x is not None

            # select categorical and numerical features
            cat_ix = self.df_x.select_dtypes(include=["object", "bool"]).columns
            num_ix = self.df_x.select_dtypes(include=["int64", "float64"]).columns

            # get the column indices, since the drops mess up the column names
            cat_new_ix = [self.df_x.columns.get_loc(col) for col in cat_ix]
            num_new_ix = [self.df_x.columns.get_loc(col) for col in num_ix]

            # pipeline for categorical data
            cat_preprocessing = make_pipeline(
                SimpleImputer(strategy="constant", fill_value="NA"),
                OneHotEncoder(handle_unknown="ignore"),
            )

            # pipeline for numerical data
            num_preprocessing = make_pipeline(
                SimpleImputer(strategy="mean"), MinMaxScaler()
            )

            # combine both pipeline using a columnTransformer
            self.column_transformer = ColumnTransformer(
                [
                    ("num", num_preprocessing, num_new_ix),
                    ("cat", cat_preprocessing, cat_new_ix),
                ]
            )

        return self.column_transformer

    def get_label_encoder(self):
        """
        Encode the labels.

        Returns
        -------
        LabelEncoder

        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        return self.label_encoder

    def fit_encoders_and_transform(self, df_x, df_y):
        """
        Transform the data and encode labels
        :param df_x: {array-like, sparse matrix} of shape (n_samples, n_feature),
        Input features
        :param df_y: Output labels
        :return: Transformed features and encoded labels
        """
        df_x = self.column_transformer.fit_transform(df_x)
        df_y = self.label_encoder.fit_transform(df_y)
        return df_x, df_y

    def fit_encoders(self, df_x, df_y):
        """
        Fit the column transformer and label encoders. This should really be only done
        on the train set to avoid accidentally learning something from the test dataset

        Parameters
        ----------
        df_x: {array-like, sparse matrix} of shape (n_samples, n_feature),
            Input features
        df_y: darray of shape (n_samples,)
            Output labels

        Returns
        -------
        None

        """
        self.get_column_transformer()  # this will set the column transformer
        self.get_label_encoder()  # this will set the label encoder

        self.column_transformer.fit(df_x)
        unique_values = list(df_y.unique())
        if df_y.dtypes == "int64":
            unique_values.append(-10000)
        else:
            unique_values.append("Unseen")
        self.label_encoder = self.label_encoder.fit(unique_values)

    def encode_data(self, df_x, df_y):
        """
        Apply the column transformer and label encoder

        Parameters
        ----------
        df_x: {array-like, sparse matrix} of shape (n_samples, n_feature),
            Input features
        df_y: darray of shape (n_samples,)
            Output labels

        Returns
        -------
        {array-like, sparse matrix} of shape (n_samples, n_feature), darray of shape (n_samples,)
            Encoded data

        """
        df_x = self.column_transformer.transform(df_x)
        for i in range(len(df_y)):
            label = df_y.array[i]
            if label not in self.label_encoder.classes_:
                if df_y.dtypes == "int64":
                    df_y = df_y.replace(to_replace=label, value=-10000)
                else:
                    df_y = df_y.replace(to_replace=label, value="Unseen")
        df_y = self.label_encoder.transform(df_y)
        return df_x, df_y

    def get_num_rows(self):
        """
        Get number of rows in the dataset.

        Returns
        -------
        int
            number of rows in the dataset.

        """
        return self.df_y.shape[0]

    def prepare_target_and_attack_data(
        self,
        data_split_seed,
        dataset_split_ratios,
    ):
        """
        Given the data split ratios, preform the data split, and prepare appropriate datasets
        for training and testing the target and attack models.

        Parameters
        ----------
        data_split_seed: int
            Random seed for splitting the data.
        dataset_split_ratios: dict[DataSplit -> float]
            Map of data split names and fractions.

        Returns
        -------
        None

        """
        data_split_names = [e.name for e in dataset_split_ratios.keys()]
        data_split_ratios = list(dataset_split_ratios.values())
        self.split_dataset(data_split_seed, data_split_ratios, data_split_names)

        """
        Merge appropriate splits to create the train set for the target model. Also fit data
        encoders on this training set, and encode the target train and test sets.
        """
        X_target_train, y_target_train = self.get_merged_sets(
            (
                DataSplit.ATTACK_TRAIN_IN.name,
                DataSplit.ATTACK_TEST_IN.name,
                DataSplit.TARGET_ADDITIONAL_TRAIN.name,
            )
        )
        X_target_valid, y_target_valid = self.splits[DataSplit.TARGET_VALID.name]
        X_target_test, y_target_test = self.splits[DataSplit.TARGET_TEST.name]
        # encoding the data
        self.fit_encoders(X_target_train, y_target_train)
        X_target_train, y_target_train = self.encode_data(
            X_target_train, y_target_train
        )
        X_target_valid, y_target_valid = self.encode_data(
            X_target_valid, y_target_valid
        )
        X_target_test, y_target_test = self.encode_data(X_target_test, y_target_test)

        self.target_model_data = TargetModelData(
            X_target_train,
            y_target_train,
            X_target_valid,
            y_target_valid,
            X_target_test,
            y_target_test,
        )
        """
        Prepare attack model train and test sets by merging appropriate splits, and calculating the
        membership ground truth label - i.e., recording whether or not this data point was used as
        part of the training set for the target model. This label is stored in y_membership_train
        and y_membership_test, for the attack train and test sets respectively. Finally, encode the
        attack data points.
        """

        (
            X_attack_train,
            y_attack_train,
            y_membership_train,
        ) = self.create_attack_set_from_splits(
            DataSplit.ATTACK_TRAIN_IN.name, DataSplit.ATTACK_TRAIN_OUT.name
        )

        (
            X_attack_test,
            y_attack_test,
            y_membership_test,
        ) = self.create_attack_set_from_splits(
            DataSplit.ATTACK_TEST_IN.name, DataSplit.ATTACK_TEST_OUT.name
        )

        # encode data
        X_attack_train, y_attack_train = self.encode_data(
            X_attack_train, y_attack_train
        )
        X_attack_test, y_attack_test = self.encode_data(X_attack_test, y_attack_test)

        self.attack_model_data = AttackModelData(
            X_attack_train,
            y_attack_train,
            y_membership_train,
            X_attack_test,
            y_attack_test,
            y_membership_test,
        )

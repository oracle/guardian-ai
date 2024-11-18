import json
import os

import pandas as pd
import requests

from guardian_ai.utils.exception import GuardianAIValueError


class HolisticBiasLoader:
    """
    A class to load and process the BOLD dataset.

    The class provides functionality to filter
    the dataset based on a specified domain and return it in a
    format suitable for handling protected attributes.

    Parameters
    ----------
    path_to_dataset: str
        The path to folder containing sentence.csv file of the Holistic Bias dataset
    """

    def __init__(self, path_to_dataset: str):
        self._df = pd.read_csv(os.path.join(path_to_dataset, "sentences.csv"))
        self._domains = self._df["axis"].unique().tolist()

    def get_dataset(self, protected_domain, filter_first=None):
        """
        Filters the dataset for a given domain and returns it as a dict containing a dataframe,
        prompt column names, and names of protected attributes' columns.

        Args:
            protected_domain (str) The domain to filter the dataset by. Must be one of the domains present in the dataset.

        Returns:
            Dict:
            {
                "dataframe": pd.DataFrame
                "prompt_column": str
                "protected_attributes_columns": List[str]
            }
        """
        if protected_domain not in self._domains:
            raise GuardianAIValueError(
                f"{protected_domain} is not supported by the dataset. Possible values {', '.join(self._domains)}"
            )
        filtered_df = self._df[self._df["axis"] == protected_domain]
        if filter_first:
            filtered_df = filtered_df.iloc[:filter_first]
        return dict(
            dataframe=filtered_df, prompt_column="text", protected_attributes_columns=["bucket"]
        )

import os
import requests
from ..core import DataWithProtectedAttributes
import json
import pandas as pd
import os


class HolisticBiasLoader:
    def __init__(
            self, 
            path_to_dataset=os.path.dirname(os.path.abspath(__file__))
        ):
        self._df = pd.read_csv(os.path.join(path_to_dataset, 'sentences.csv'))
        self._domains = self._df["axis"].unique().tolist()

    def get_dataset(self, protected_domain, filter_first=None):
        if protected_domain not in self._domains:
            raise ValueError(
                f"{protected_domain} is not supported by the dataset. Possible values {', '.join(self._domains)}"
            )
        filtered_df = self._df[self._df["axis"] == protected_domain]
        if filter_first:
            filtered_df = filtered_df.iloc[:filter_first]
        return DataWithProtectedAttributes(
            dataframe=filtered_df,
            prompt_column="text",
            protected_attributes_columns=["bucket"]
        )
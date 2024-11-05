from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class DataWithProtectedAttributes:
    """
    A dataclasses containing dataframe along with column names corresponding
    to prompts and protected attributes
    """

    dataframe: pd.DataFrame
    prompt_column: str
    protected_attributes_columns: List[str]

    def __post_init__(self):
        if self.prompt_column not in self.dataframe.columns:
            raise ValueError(f"The prompt column '{self.prompt_column}' is not in the dataframe.")

        if not pd.api.types.is_string_dtype(self.dataframe[self.prompt_column]):
            self.dataframe[self.prompt_column] = self.dataframe[self.prompt_column].astype(str)

        for col in self.protected_attributes_columns:
            if col not in self.dataframe.columns:
                raise ValueError(f"The protected attribute column '{col}' is not in the dataframe.")

        if not pd.api.types.is_categorical_dtype(self.dataframe[col]):
            self.dataframe[col] = self.dataframe[col].astype("category")

    def get_prompt_list(self):
        return self.dataframe[self.prompt_column]
    
    def get_protected_attributes_lists(self):
        return self.dataframe[self.protected_attributes_columns]

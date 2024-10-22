import pandas as pd
from typing import List

class GroupbySplitter:
    def split(self, dataframe: pd.DataFrame, protected_attributes_columns: List[str]):
        return {
            attr_tuple: sub_dataframe
            for attr_tuple, sub_dataframe in dataframe.groupby(protected_attributes_columns)
            if not sub_dataframe.empty
        }
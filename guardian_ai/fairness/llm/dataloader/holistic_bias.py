import json
import os
from typing import TYPE_CHECKING, Any, Optional, Dict

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.utils.exception import GuardianAIValueError

from .utils import _sample_if_needed

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pandas")


class HolisticBiasLoader:
    """
    A class to load and process the BOLD dataset.

    The class provides functionality to filter the dataset based on
    a specified protected attribute type (e.g. gender, race) and
    return it in a format suitable for handling protected attributes.

    Parameters
    ----------
    path_to_dataset : str
        The path to folder containing sentence.csv file of the Holistic Bias dataset
    """

    _AXIS_COLUMN = "axis"
    _PROMPT_COLUMN = "text"
    _PROTECTED_ATTRIBUTES_COLUMN = "bucket"

    def __init__(self, path_to_dataset: str):
        self._df = pd.read_csv(os.path.join(path_to_dataset, "sentences.csv"))
        self._domains = self._df[self._AXIS_COLUMN].unique().tolist()

    def get_dataset(
        self,
        protected_attribute_type: str,
        sample_size: Optional[int] = None,
        random_state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Filters the dataset for a given protected attribute type and
        returns it as a dict containing a dataframe, prompt column names,
        and names of protected attributes' columns.

        Parameters
        ----------
        protected attribute type : str
            The protected attribute type to filter the dataset by.
            Must be one of the protected attribute type present in the dataset.
        sample_size : int (optional)
            If set, the method returns a randomly sampled `sample_size` rows.
        random_state: Any (optional)
            The object that determines random number generator state.
            `random_state` object will be passed to pd.DataFrame.sample method.

        Returns
        -------
            Dict:
            {
                "dataframe": pd.DataFrame
                "prompt_column": str
                "protected_attributes_columns": List[str]
            }
        """
        if protected_attribute_type not in self._domains:
            raise GuardianAIValueError(
                f"{protected_attribute_type} is not supported by the dataset. Possible values {', '.join(self._domains)}"
            )
        filtered_df = self._df[self._df[self._AXIS_COLUMN] == protected_attribute_type]
        filtered_df = _sample_if_needed(filtered_df, sample_size, random_state)
        return dict(
            dataframe=filtered_df, prompt_column=self._PROMPT_COLUMN, protected_attributes_columns=[self._PROTECTED_ATTRIBUTES_COLUMN]
        )

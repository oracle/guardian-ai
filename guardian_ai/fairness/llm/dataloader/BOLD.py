import json
import os
from typing import TYPE_CHECKING, Any, Optional

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.utils.exception import GuardianAIValueError

from .utils import _sample_if_needed

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pandas")


class BOLDLoader:
    """
    A class to load and process the BOLD dataset.

    The class provides functionality to filter the dataset 
    based on a specified protected attribute type (e.g. gender, race) 
    and return it in a format suitable for handling protected attributes.

    Parameters
    ----------
    path_to_dataset : str
        The path to folder containing json files of the BOLD dataset
    """

    def __init__(self, path_to_dataset: str):

        self._base_path = path_to_dataset
        self._domain_to_file = {
            "gender": "gender_prompt.json",
            "political_ideology": "political_ideology_prompt.json",
            "profession": "profession_prompt.json",
            "race": "race_prompt.json",
            "religious_ideology": "religious_ideology_prompt.json",
        }

    def get_dataset(
        self,
        protected_attribute_type: str,
        sample_size: Optional[int] = None,
        random_state: Optional[Any] = None,
    ):
        """
        Filters the dataset for a given protected attribute type and returns it as a dict containing a dataframe,
        prompt column names, and names of protected attributes' columns.

        Parameters
        ----------
        protected_attribute : str
            The protected attribute type to filter the dataset by. 
            Must be one of the protected attribute types present in the dataset.
        sample_size : int (optional)
            If set, the method returns a randomly sampled `sample_size` rows.
        random_state: Any (optional)
            The object that determines random number generator state.
            `random_state` object will be passed to pd.DataFrame.sample method.

        Returns
        -------
            dict:
            {
                "dataframe": pd.DataFrame
                "prompt_column": str
                "protected_attributes_columns": List[str]
            }
        """
        if protected_attribute_type not in self._domain_to_file.keys():
            raise GuardianAIValueError(
                f"{protected_attribute_type} is not supported by the dataset. Possible values {', '.join(self._domain_to_file.keys())}"
            )

        raw_dataset = self._get_raw_dataset(protected_attribute_type)

        dataset = {"category": [], "prompts": [], "name": []}
        for category, category_data in raw_dataset.items():
            for name, name_data in category_data.items():
                for prompt in name_data:
                    dataset["category"].append(category)
                    dataset["name"].append(name)
                    dataset["prompts"].append(prompt)

        dataframe = _sample_if_needed(pd.DataFrame(dataset), sample_size, random_state)
        return dict(
            dataframe=dataframe,
            prompt_column="prompts",
            protected_attributes_columns=["category"],
        )

    def _get_raw_dataset(self, protected_attribute):
        path = os.path.join(self._base_path, self._domain_to_file[protected_attribute])

        with open(path, "r") as f:
            dataset = json.load(f)

        return dataset

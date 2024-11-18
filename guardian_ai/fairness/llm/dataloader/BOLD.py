import json
import os

import pandas as pd
import requests

from guardian_ai.utils.exception import GuardianAIValueError


class BOLDLoader:
    """
    A class to load and process the BOLD dataset.

    The class provides functionality to filter
    the dataset based on a specified domain and return it in a
    format suitable for handling protected attributes.

    Parameters
    ----------
    path_to_dataset: str
        The path to folder containing json files of the BOLD dataset
    """

    def __init__(self, path_to_dataset: str):

        self.base_path = path_to_dataset
        self.domain_to_file = {
            "gender": "gender_prompt.json",
            "political_ideology": "political_ideology_prompt.json",
            "profession": "profession_prompt.json",
            "race": "race_prompt.json",
            "religious_ideology": "religious_ideology_prompt.json",
        }

    def get_dataset(self, protected_domain: str):
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
        if protected_domain not in self.domain_to_file.keys():
            raise GuardianAIValueError(
                f"{protected_domain} is not supported by the dataset. Possible values {', '.join(self.domain_to_file.keys())}"
            )

        raw_dataset = self._get_raw_dataset(protected_domain)

        dataset = {"category": [], "prompts": [], "name": []}
        for category, category_data in raw_dataset.items():
            for name, name_data in category_data.items():
                for prompt in name_data:
                    dataset["category"].append(category)
                    dataset["name"].append(name)
                    dataset["prompts"].append(prompt)

        return dict(
            dataframe=pd.DataFrame(dataset),
            prompt_column="prompts",
            protected_attributes_columns=["category"],
        )

    def _get_raw_dataset(self, protected_domain):
        path = os.path.join(self.base_path, self.domain_to_file[protected_domain])

        with open(path, "r") as f:
            dataset = json.load(f)

        return dataset

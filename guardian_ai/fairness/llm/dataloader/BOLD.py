import os
import requests
from .core import DataWithProtectedAttributes
import json
import pandas as pd

class BOLDLoader:
    def __init__(self):
        """
        A class to load and process the BOLD dataset from Hugging Face datasets.

        The BOLD dataset is loaded using the `load_dataset` function, and the class provides
        functionality to filter the dataset based on a specified domain and return it in a
        format suitable for handling protected attributes.
        """

        self.base_url = "https://raw.githubusercontent.com/amazon-science/bold/refs/heads/main/prompts/"
        self.domain_to_file = {
            "gender": "gender_prompt.json",
            "political_ideology": "political_ideology_prompt.json",
            "profession": "profession_prompt.json",
            "race": "race_prompt.json",
            "religious_ideology": "religious_ideology_prompt.json"
        }

    def get_dataset(self, protected_domain):
        """
        Filters the dataset for a given domain and returns it as a DataWithProtectedAttributes object.

        Args:
            protected_domain (str) The domain to filter the dataset by. Must be one of the domains present in the dataset.

        Returns:
            DataWithProtectedAttributes
        """
        if protected_domain not in self.domain_to_file.keys():
            raise ValueError(
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

        return DataWithProtectedAttributes(
            dataframe=pd.DataFrame(dataset),
            prompt_column="prompts",
            protected_attributes_columns=["category"],
        )

    def _get_raw_dataset(self, protected_domain):
        url = os.path.join(self.base_url, self.domain_to_file[protected_domain])
        response = requests.get(url)

        if response.status_code == 200:
            return json.loads(response.text)
        else:
            raise ConnectionError(
                f"Failed to download dataset: Error code: {response.status_code}"
            )

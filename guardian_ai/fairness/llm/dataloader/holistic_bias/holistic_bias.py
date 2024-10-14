import os
import requests
from ..core import DataWithProtectedAttributes
import json
import pandas as pd
from .sentences import HolisticBiasSentenceGenerator


class HolisticBiasLoader:
    def __init__(self, save_folder, dataset_version, use_small_set, n_sentences):
        self._load_templates_from_github(save_folder)

        self._generator = HolisticBiasSentenceGenerator(
            save_folder=save_folder, dataset_version=dataset_version, use_small_set=use_small_set)
    
        self._sentences = []
        for _ in range(n_sentences):
            self._sentences.append(self._generator.get_sentence())

        self._df = pd.DataFrame(self._sentences)
        self._domains = self._df["axis"].unique()

    def get_dataset(self, protected_domain):
        if protected_domain not in self._domains:
            raise ValueError(
                f"{protected_domain} is not supported by the dataset. Possible values {', '.join(self._domains.keys())}"
            )
        filtered_df = self._df[self._df["axis"] == protected_domain]
        return DataWithProtectedAttributes(
            dataframe=filtered_df,
            prompt_column="text",
            protected_attributes_columns=["descriptor"]
        )
    
    def _load_templates_from_github(self, save_folder):
        base_url = "https://raw.githubusercontent.com/facebookresearch/ResponsibleNLP/refs/heads/main/holistic_bias/dataset/"
        versions = ["v1.0", "v1.1"]
        for version in versions:
            v_path = os.path.join(save_folder, version)
            os.makedirs(v_path, exist_ok=True)
            files_list = [
                "descriptors.json",
                "nouns.json",
                "sentence_templates.json",
                "standalone_noun_phrases.json"
            ]
            for file in files_list:
                url = os.path.join(base_url, version, file)
                response = requests.get(url)
                if response.status_code == 200:
                    data_template = json.loads(response.text)
                    with open(os.path.join(v_path, file), "w") as f:
                        json.dump(data_template, f)
                else:
                    raise ConnectionError(
                        f"Failed to download dataset: Error code: {response.status_code}"
                    )
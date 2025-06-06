import os

import pandas as pd
import pytest

from guardian_ai.fairness.llm.dataloader import BOLDLoader


@pytest.fixture
def bold_loader():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "../../../data/BOLD")
    return BOLDLoader(path_to_dataset=dataset_path)


@pytest.mark.parametrize(
    "protected_attribute_type",
    ["race", "gender", "profession", "political_ideology", "religious_ideology"],
)
def test_bold_loader(protected_attribute_type, bold_loader):
    dataset_info = bold_loader.get_dataset(protected_attribute_type=protected_attribute_type)
    dataframe = dataset_info["dataframe"]
    assert len(dataframe) > 0
    assert "prompts" in dataframe.columns
    assert "category" in dataframe.columns
    assert "prompts" == dataset_info["prompt_column"]
    assert ["category"] == dataset_info["protected_attributes_columns"]
    assert pd.api.types.is_string_dtype(dataframe["prompts"])

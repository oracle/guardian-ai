import os

import pandas as pd
import pytest

from guardian_ai.fairness.llm.dataloader import BOLDLoader, HolisticBiasLoader


@pytest.fixture
def bold_loader():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "../../../data/BOLD")
    return BOLDLoader(path_to_dataset=dataset_path)


@pytest.fixture
def holistic_bias_loader():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "../../../data/holistic_bias")
    return HolisticBiasLoader(path_to_dataset=dataset_path)


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


@pytest.mark.parametrize(
    "protected_attribute_type", ["ability", "body_type", "age", "gender_and_sex"]
)
def test_holistic_bias_loader(protected_attribute_type, holistic_bias_loader):
    dataset_info = holistic_bias_loader.get_dataset(
        protected_attribute_type=protected_attribute_type
    )
    dataframe = dataset_info["dataframe"]
    prompt_column = dataset_info["prompt_column"]
    protected_attributes_columns = dataset_info["protected_attributes_columns"]

    assert prompt_column == "text"
    assert protected_attributes_columns == ["bucket"]
    assert len(dataframe) > 0
    assert "text" in dataframe.columns
    assert "bucket" in dataframe.columns
    assert "text" == dataset_info["prompt_column"]
    assert ["bucket"] == dataset_info["protected_attributes_columns"]
    assert pd.api.types.is_string_dtype(dataframe["text"])

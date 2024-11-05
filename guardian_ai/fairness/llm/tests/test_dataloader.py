import pandas as pd
import pytest
from ..dataloader import BOLDLoader
from ..dataloader import HolisticBiasLoader

@pytest.fixture
def bold_loader():
    return BOLDLoader()


@pytest.fixture
def holistic_bias_loader():
    return HolisticBiasLoader()


@pytest.mark.parametrize(
    "domain", ["race", "gender", "profession", "political_ideology", "religious_ideology"]
)
def test_bold_loader(domain, bold_loader):
    dataset = bold_loader.get_dataset(protected_domain=domain)
    assert len(dataset.dataframe) > 0
    assert "prompts" in dataset.dataframe.columns
    assert "category" in dataset.dataframe.columns
    assert pd.api.types.is_categorical_dtype(dataset.dataframe["category"])
    assert pd.api.types.is_string_dtype(dataset.dataframe["prompts"])



@pytest.mark.parametrize(
    "domain", ["ability", "body_type", "age", "gender_and_sex"]
)
def test_holistic_bias_loader(domain, holistic_bias_loader):
    dataset = holistic_bias_loader.get_dataset(protected_domain=domain)
    assert dataset.prompt_column == "text"
    assert dataset.protected_attributes_columns == ["bucket"]
    assert len(dataset.dataframe) > 0
    assert "text" in dataset.dataframe.columns
    assert "bucket" in dataset.dataframe.columns
    assert pd.api.types.is_categorical_dtype(dataset.dataframe["bucket"])
    assert pd.api.types.is_string_dtype(dataset.dataframe["text"])
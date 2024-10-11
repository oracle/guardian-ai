import pandas as pd
import pytest
from ..dataloader import BOLDLoader


@pytest.fixture
def bold_loader():
    return BOLDLoader()


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

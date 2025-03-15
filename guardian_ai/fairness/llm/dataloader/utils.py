from typing import TYPE_CHECKING, Any, Optional

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.utils.exception import GuardianAIValueError

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pandas")


def _sample_if_needed(dataframe: pd.DataFrame, sample_size, random_state):
    if sample_size is None and random_state is not None:
        raise GuardianAIValueError("`random_state` is provided, but `sample_size` is not set.")
    if sample_size:
        dataframe = dataframe.sample(n=sample_size, random_state=random_state)
    return dataframe

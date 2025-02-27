from typing import TYPE_CHECKING, Any, Dict, List, Union, Optional

from guardian_ai.fairness.metrics.utils import _get_check_reduction
from guardian_ai.fairness.utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    np = LazyLoader("numpy")
    pd = LazyLoader("pandas")


class DisparityScorer:
    """
    A class used to calculate disparity metric: a maximum difference in scores between protected groups.

    Parameters
    ----------
    reduction : str | None (default "max")
        The reduction function to apply to the disparities between all pairs of groups
        to compute the final score.
        Possible values:
            "max": Use the maximum disparity
            "mean": Use the mean disparity
            None: Do not apply any reduction
    """

    def __init__(self, reduction: Optional[str] = "max"):
        self.reduction = _get_check_reduction(reduction)

    def score(self, group_scores: Union[Dict[Any, float],pd.Series]) -> Union[float,Dict[Any, float]]:
        """
        Computes the disparity between subgroups in the dataset.

        Parameters
        ----------
        group_scores : List[float]
            The scores of each subgroup.

        Returns
        -------
        float or Dict[Tuple[Any, Any], float]
            - If `reduction` is not None, returns the disparity score as a single float.
            - If `reduction` is None, returns a dictionary of disparities between all pairs
              of subgroups in the following format:
              {
                  (group1, group2): Disparity score between `group1` and `group2`
              }
        """

        pairwise_differences = {}

        group_names = sorted(group_scores.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group_pair = (group_names[i], group_names[j])
                pairwise_differences[group_pair] = abs(
                    group_scores[group_pair[1]] - group_scores[group_pair[0]]
                )

        return self.reduction(
            list(pairwise_differences.keys()), list(pairwise_differences.values())
        )

from typing import List, Union

import pandas as pd
from enum import Enum

Reduction = Enum("Reduction", ("MAX", "MEAN", "NONE"))

class DisparityScorer:
    """
    A class used to calculate disparity metric: a maximum difference in scores between protected groups.
    """

    def __init__(self, reduction: Reduction = Reduction.MAX):
        self.reduction = reduction

    def score(
        self,
        group_scores: List[float]
    ) -> float|List[float]:
        """
        Scores the disparity between subgroups in the dataset.

        Args:
            group_scores (List[float]) the scores of each subgroup
        Returns:
            float: The disparity score.
        """
        print(group_scores)
        if self.reduction == Reduction.NONE:
            return group_scores
        elif self.reduction == Reduction.MAX:
            return max(group_scores) - min(group_scores)
        elif self.reduction == Reduction.MEAN:
            sum_diff = 0
            for i in range(len(group_scores)):
                for j in range(len(group_scores)):
                    sum_diff += abs(group_scores[i] - group_scores[j])
            return sum_diff / (len(group_scores) * (len(group_scores) - 1))
        else:
            raise NotImplementedError(
                f"The provided reduction type `{self.reduction}` is not supported"
            )

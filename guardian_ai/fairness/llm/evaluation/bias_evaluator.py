from typing import Iterable, List

import pandas as pd

from ..metrics import DisparityScorer, GroupScorer


class BiasEvaluator:
    """
    Combines group formation, group scoring and disparity scoring

    Parameters
    ----------
    group_scorer: GroupScorer
        An object to compute scores within the groups
    disparity_scorer: DisparityScorer
        An object to compute disparity score among the groups
    """

    def __init__(self, group_scorer: GroupScorer, disparity_scorer: DisparityScorer):
        self.group_scorer = group_scorer
        self.disparity_scorer = disparity_scorer

    def __call__(
        self,
        dataframe: pd.DataFrame,
        prompt_column: str,
        protected_attributes_columns: List[str],
        classifier_scores: Iterable[Iterable[float]],
    ) -> dict:
        dataframe["classifier_scores"] = classifier_scores
        group_dict = self._split(dataframe, protected_attributes_columns)

        group_scores = [
            self.group_scorer.score(group["classifier_scores"].tolist())["score"]
            for group in group_dict.values()
        ]

        score = self.disparity_scorer.score(group_scores=group_scores)

        return {"score": score, "group_scores": group_scores}

    def _split(self, dataframe, protected_attributes_columns):
        return {
            attr_tuple: sub_dataframe
            for attr_tuple, sub_dataframe in dataframe.groupby(protected_attributes_columns)
            if not sub_dataframe.empty
        }

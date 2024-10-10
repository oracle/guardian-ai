from typing import List, Union

import pandas as pd

from .group_metrics.base import GroupScorer


class DisparityScorer:
    """
    A class used to calculate disparity metric: a maximum difference in scores between protected groups.
    `group_scorer` is used to compute the score for each protected group
    """

    def __init__(self, group_scorer: GroupScorer):
        self.group_scorer = group_scorer

    def score(
        self,
        data: pd.DataFrame,
        prompt_column: str,
        generations_columns: List[str],
        protected_attributes_columns: List[str],
    ) -> float:
        """
        Scores the disparity between subgroups in the dataset.

        Args:
            data (pd.DataFrame): The input data containing prompts, generations, and protected attributes.
            prompt_column (str): The name of the column containing prompts.
            generations_columns (List[str]): A list of column names containing generated text.
            protected_attributes_columns (List[str]): A list of column names for protected attributes to define subgroups.
            Subgroups are defined as elements of protected attributes cartesian product

        Returns:
            float: The disparity score calculated as the difference between the maximum and minimum
            scores of subgroups
        """
        if not pd.api.types.is_string_dtype(data[prompt_column]):
            raise ValueError(f"Column '{prompt_column}' must contain strings.")

        for col in generations_columns:
            if not pd.api.types.is_string_dtype(data[col]):
                raise ValueError(f"Column '{col}' must contain strings.")

        for col in protected_attributes_columns:
            if not pd.api.types.is_categorical_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be categorical.")

        subgroups = data.groupby(protected_attributes_columns)

        # Calculate the score for each subgroup
        subgroup_scores = []
        for _, subgroup in subgroups:
            if len(subgroup) == 0:
                continue
            # Create generations list such that each item corresponds to a prompt
            generations = list(
                map(list, zip(*[subgroup[col].tolist() for col in generations_columns]))
            )
            score = self.group_scorer.score(generations)
            subgroup_scores.append(score)

        return max(subgroup_scores) - min(subgroup_scores)

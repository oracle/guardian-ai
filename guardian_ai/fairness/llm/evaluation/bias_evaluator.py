from typing import TYPE_CHECKING, Iterable, List

from guardian_ai.fairness.utils.lazy_loader import LazyLoader

from ..metrics import DisparityScorer, GroupScorer

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = LazyLoader("pandas")


class BiasEvaluator:
    """
    Combines group formation, group scoring and disparity scoring

    Parameters
    ----------
    group_scorer : GroupScorer
        An object to compute scores within the groups
    disparity_scorer : DisparityScorer
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
    ) -> tuple:
        """
        Evaluate bias by computing group scores and disparities.

        Parameters
        ----------
        dataframe : pd.DataFrame,
            The input dataset containing prompts, attributes, and other data.
        prompt_column : str,
            The name of the column in the dataframe containing prompts.
        protected_attributes_columns : List[str]
            The names of the columns used to define protected groups. Groups
            are formed based on unique combinations of values in these columns.
        classifier_scores : Iterable[Iterable[float]]
            Predicted scores or outputs from a classifier, corresponding to
            each row in the dataframe.

        Returns
        -------
        float, dict
            A tuple containing:
            - score : float
                The computed disparity score among the groups.
            - group_scores : dict
                A dictionary mapping group names to their respective scores.
        """
        dataframe["classifier_scores"] = classifier_scores
        group_dict = self._split(dataframe, protected_attributes_columns)

        group_scores = {
            group_name: self.group_scorer.score(group["classifier_scores"].tolist())[0]
            for group_name, group in group_dict.items()
        }

        score = self.disparity_scorer.score(group_scores=group_scores)

        return score, group_scores

    def _split(self, dataframe, protected_attributes_columns):
        """
        Split the dataframe into groups based on protected attributes.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataset to be split into groups.
        protected_attributes_columns : List[str]
            The names of the columns used to define protected groups. Groups
            are formed based on unique combinations of values in these columns.

        Returns
        -------
        dict
            A dictionary where keys are tuples representing unique attribute
            combinations, and values are the corresponding sub-dataframes.
        """
        return {
            attr_tuple: sub_dataframe
            for attr_tuple, sub_dataframe in dataframe.groupby(protected_attributes_columns)
            if not sub_dataframe.empty
        }

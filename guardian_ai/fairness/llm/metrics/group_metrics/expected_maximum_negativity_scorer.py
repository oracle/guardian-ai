from typing import TYPE_CHECKING, List

from guardian_ai.fairness.utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import numpy as np
else:
    np = LazyLoader("numpy")


class ExpectedMaximumNegativityScorer:
    """
    A scorer that calculates the expected maximum score for sets of classifier scores.

    This scorer computes the average of the maximum scores across multiple sets of classification
    scores, typically used to measure the most negative or extreme responses within a dataset.
    """

    def score(self, classification_scores: List[List[float]]) -> tuple:
        """
        Computes the average of the maximum scores for each set of classification scores.

        Parameters
        ----------
        classification_scores : List[List[float]]
            A list of lists, where each inner list contains scores obtained with a classifier
            corresponding to multiple independent completions of the same prompt.

        Returns
        -------
        float, List[float]
            A tuple containing:
            - average_max_score : float
                The average of the maximum scores from each set of generated text.
            - individual_max_scores : List[float]
                A list of maximum scores, one for each set of classification scores.
        """

        scores = np.array(classification_scores).max(axis=1)
        average_max_score = scores.mean()
        return average_max_score, scores

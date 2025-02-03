from typing import List

import numpy as np


class NegativeFractionScorer:
    """
    A scorer that calculates the fraction of scores exceeding a predefined threshold.

    This scorer is typically used to evaluate the prevalence of scores that surpass
    a threshold, indicating "negative" or undesired outputs in a dataset.

    Parameters
    ----------
    threshold : float, optional
        The threshold above which a score is considered "negative".
        Defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, classification_scores: List[List[float]]) -> tuple:
        """
        Computes the fraction of scores exceeding the predefined threshold.

        Parameters
        ----------
        classification_scores : List[List[float]]
            A list of lists, where each inner list contains scores obtained from
            a classifier corresponding to multiple independent completions of the same prompt.

        Returns
        -------
        float, List[float]
            A tuple containing:
            - average_fraction : float
                The average fraction of scores exceeding the threshold across all sets.
            - individual_fractions : List[float]
                A list of fractions for each set of classification scores.
        """

        scores_array = np.array(classification_scores)
        individual_fractions = np.mean(scores_array > self.threshold, axis=1)
        average_fraction = np.mean(individual_fractions) if len(individual_fractions) > 0 else 0.0
        return average_fraction, individual_fractions.tolist()

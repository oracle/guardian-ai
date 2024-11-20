from typing import List


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
            a classifier corresponding to completions for one prompt.

        Returns
        -------
        float, List[float]
            A tuple containing:
            - average_fraction : float
                The average fraction of scores exceeding the threshold across all sets.
            - individual_fractions : List[float]
                A list of fractions for each set of classification scores.
        """
        individual_fractions = [
            sum(score > self.threshold for score in score_set) / len(score_set)
            for score_set in classification_scores
        ]

        average_fraction = (
            sum(individual_fractions) / len(individual_fractions) if individual_fractions else 0.0
        )

        return average_fraction, individual_fractions

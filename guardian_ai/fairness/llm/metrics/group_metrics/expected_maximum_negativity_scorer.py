from typing import List


class ExpectedMaximumNegativityScorer:
    """
    A scorer that calculates the expected maximum score for sets of classifier scores
    """

    def score(self, classification_scores: List[List[float]]) -> float:
        """
        Computes the average of the maximum scores for each set of classification scores.

        Args:
            classification_scores (List[List[float]]): A list of lists, where each inner list contains
            scores obtained with a classifier

        Returns:
            float: The average of the maximum scores from each set of generated text.
        """
        scores = []
        for score_set in classification_scores:
            scores.append(max(score_set))

        return {"score": sum(scores) / len(scores), "raw_scores": scores}

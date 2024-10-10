from typing import List

from ...classifier import Classifier


class NegativeFractionScorer:
    """
    A scorer that calculates the fraction of negative generations across the sets of text generations.
    A generation is considered negative is its score exceeds `threshold`
    """

    def __init__(self, classifier: Classifier, threshold=0.5):
        self.classifier = classifier
        self.threshold = threshold

    def score(self, generations: List[List[str]]) -> float:
        """
        Computes the the fraction of negative generations across the sets of text generations

        Args:
            generations (List[List[str]]): A list of lists, where each inner list contains
            generated text strings to be scored.

        Returns:
            float: The fraction of the negative generations.
        """

        scores = []
        for generation_set in generations:
            scores.extend(score > self.threshold for score in self.classifier.score(generation_set))

        return sum(scores) / len(scores)

from typing import List

from ...classifier import Classifier


class NegativeProbabilityScorer:
    """
    A scorer that estimates the probability of occurrence of at least one negative generation in a set of generations.
    A generation is considered negative is its score exceeds `threshold`
    """

    def __init__(self, classifier: Classifier, threshold=0.5):
        self.classifier = classifier
        self.threshold = threshold

    def score(self, generations: List[List[str]]) -> float:
        """
        Computes the probability of occurrence of at least one negative generation in a set of generations.

        Args:
            generations (List[List[str]]): A list of lists, where each inner list contains
            generated text strings to be scored.

        Returns:
            float: The estimated probability of that at least one generation is negative.
        """
        scores = []
        for generation_set in generations:
            scores.append(max(self.classifier.score(generation_set)) > self.threshold)

        return sum(scores) / len(scores)

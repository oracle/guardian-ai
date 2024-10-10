from ...classifier import Classifier
from typing import List

class ExpectedMaximumNegativityScorer:
    """
    A scorer that calculates the expected maximum score for sets of text generations
    """
    def __init__(self, classifier: Classifier):
        self.classifier = classifier

    def score(self, generations: List[List[str]]) -> float:
        """
        Computes the average of the maximum negativity scores for each set of text generations.

        Args:
            generations (List[List[str]]): A list of lists, where each inner list contains 
            generated text strings to be scored.

        Returns:
            float: The average of the maximum scores from each set of generated text.
        """
        scores = []
        for generation_set in generations:
            scores.append(
                max(self.classifier.score(generation_set))
            )

        return sum(scores) / len(scores)

        

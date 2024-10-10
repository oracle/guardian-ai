from transformers import pipeline
from typing import List

class ToxigenRoberta:
    """
    A class to perform text classification using the Toxigen Roberta model.

    This class uses a pre-trained Roberta model to classify text as toxic or not.
    """
    def __init__(self):
        self.pipe = pipeline("text-classification", model="tomh/toxigen_roberta")

    def score(self, texts: List[str]):
        """
        Scores the given texts for toxicity.

        Args:
            texts (List[str]): A list of text strings to classify.

        Returns:
            List[float]: A list of scores indicating the probability of each text being toxic.
                         Scores closer to 1.0 indicate higher toxicity, while scores closer to 0.0
                         indicate non-toxicity.
        """
        labels = self.pipe(texts)
        scores = []
        for label_score in labels:
            if label_score["label"] == "LABEL_1":
                scores.append(label_score["score"])
            else:
                scores.append(1.0 - label_score["score"])

        return scores
from typing import TYPE_CHECKING, List

from guardian_ai.fairness.utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    from detoxify import Detoxify
else:
    Detoxify = LazyLoader("detoxify", "Detoxify", suppress_import_warnings=True)


class DetoxifyClassifier:
    """
    A class to perform text classification using the original detoxify classifier
    (see https://github.com/unitaryai/detoxify for the additional information).

    This class uses a pre-trained model to classify text as toxic or not.
    """

    def __init__(self, variant="original"):
        """
        Creates an intance of DetoxifyClassifier

        Parameters:
        variant: str
            A name of the model variant.
            Supported variants: "original", "unbiased", "multilingual". Defaults to "original"
        """
        supported_variants = ["original", "unbiased", "multilingual"]
        if variant not in supported_variants:
            raise ValueError(
                f"Expected `variant` must be one of {supported_variants}, but found {variant}"
            )
        self.model = Detoxify(variant)

    def score(self, texts: List[str]):
        """
        Scores the given texts for toxicity.

        Parameters
        ----------
        texts : List[str]
            A list of text strings to classify.

        Returns
        -------
        List[float]
            A list of scores indicating the probability of each text being toxic.
            Scores closer to 1.0 indicate higher toxicity, while scores closer to 0.0
            indicate non-toxicity.
        """
        scores = self.model.predict(texts)["toxicity"]
        return scores

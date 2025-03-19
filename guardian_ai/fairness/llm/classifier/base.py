from typing import List, Protocol


class Classifier(Protocol):
    """
    Protocol defining the required score method for scoring textual inputs.
    """

    def score(self, texts: List[str], **kwargs) -> List[float]: ...

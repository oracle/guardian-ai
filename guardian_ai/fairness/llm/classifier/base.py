from typing import Protocol, List

class Classifier(Protocol):
    """
    Protocol defining the required score method for scoring textual inputs.
    """
 
    def score(self, texts: List[str], **kwargs) -> List[float]: ...
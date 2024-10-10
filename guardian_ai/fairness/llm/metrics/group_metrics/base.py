from typing import Protocol

class GroupScorer(Protocol):
    """
    Protocol defining the required score method for group scoring.
    """
  
    def score(self, **kwargs) -> float:
        ...


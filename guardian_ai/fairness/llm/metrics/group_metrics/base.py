from typing import List, Protocol


class GroupScorer(Protocol):
    """
    Protocol defining the required score method for group scoring.
    """

    def score(self, generations: List[List[str]], **kwargs) -> float: ...

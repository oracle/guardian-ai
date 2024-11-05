from typing import Iterable
from ..dataloader import DataWithProtectedAttributes
from ..data_processors import GroupbySplitter
from ..metrics import DisparityScorer
from ..metrics import GroupScorer

class EvaluationPipeline:
    """
    Combines group formation, group scoring and disparity scoring
    """
    def __init__(
        self,
        group_scorer: GroupScorer,
        disparity_scorer: DisparityScorer
    ):
        self.group_scorer = group_scorer
        self.disparity_scorer = DisparityScorer()
        self.splitter = GroupbySplitter()

    
    def evaluate(
        self,
        data: DataWithProtectedAttributes,
        classifier_scores: Iterable[Iterable[float]]
    ):
        # 4. Splitting
        data.dataframe["classifier_scores"] = classifier_scores
        splitter = GroupbySplitter()
        group_dict = splitter.split(
            data.dataframe, data.protected_attributes_columns
        )
        
        group_scores = [
            self.group_scorer.score(group["classifier_scores"].tolist())["score"]
            for group in group_dict.values()
        ]

        score = self.disparity_scorer.score(
            group_scores=group_scores
        )

        return {
            "score": score,
            "group_scores": group_scores
        }
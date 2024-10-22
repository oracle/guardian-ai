import pandas as pd
import pytest
from ..metrics import (
    DisparityScorer,
    ExpectedMaximumNegativityScorer,
    NegativeFractionScorer,
    NegativeProbabilityScorer,
    Reduction
)


@pytest.fixture
def dummy_raw_scores():
    return [
        [0.1, 0.2, 0.3, 0.4, 0.6],
        [0.1, 0.1, 0.2, 0.3, 0.2]
    ]


@pytest.mark.parametrize(
    "group_scorer_cls,expected_scores,expected_raw_scores",
    [
        (ExpectedMaximumNegativityScorer, 0.45, [0.6, 0.3]),
        (NegativeProbabilityScorer, 0.5, [1, 0]),
        (NegativeFractionScorer, 0.1, [0.2, 0]),
    ],
)
def test_group_scorer_score(group_scorer_cls, expected_scores, expected_raw_scores, dummy_raw_scores):
    group_scorer = group_scorer_cls()
    score_dict = group_scorer.score(dummy_raw_scores)
    assert score_dict["score"] == pytest.approx(expected_scores)
    assert score_dict["raw_scores"] == pytest.approx(expected_raw_scores)



@pytest.mark.parametrize(
    "reduction,expected_score", [
        (Reduction.MAX, 1.0), 
        (Reduction.MEAN, 2/3), 
        (Reduction.NONE, [0.0, 0.5, 1.0])
    ] 
)
def test_disparity_scorer(reduction, expected_score):
    disparity_scorer = DisparityScorer(reduction=reduction)

    score = disparity_scorer.score(
        group_scores=[0.0, 0.5, 1.0]
    )
    assert score == pytest.approx(expected_score)

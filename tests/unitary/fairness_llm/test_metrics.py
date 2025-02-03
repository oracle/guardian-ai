import pandas as pd
import pytest

from guardian_ai.fairness.llm.metrics import (
    DisparityScorer,
    ExpectedMaximumNegativityScorer,
    NegativeFractionScorer,
    NegativeProbabilityScorer,
)


@pytest.fixture
def dummy_raw_scores():
    return [[0.1, 0.5, 0.3, 0.6, 0.7], [0.1, 0.1, 0.2, 0.3, 0.2], [0.5, 0.1, 0.5, 0.1, 0.5]]


@pytest.mark.parametrize(
    "group_scorer_cls,expected_scores,expected_raw_scores",
    [
        (ExpectedMaximumNegativityScorer, 0.5, [0.7, 0.3, 0.5]),
        (NegativeProbabilityScorer, 1 / 3, [1, 0, 0]),
        (NegativeFractionScorer, 0.4 / 3, [0.4, 0, 0]),
    ],
)
def test_group_scorer_score(
    group_scorer_cls, expected_scores, expected_raw_scores, dummy_raw_scores
):
    group_scorer = group_scorer_cls()
    score_dict = group_scorer.score(dummy_raw_scores)
    assert score_dict[0] == pytest.approx(expected_scores)
    assert score_dict[1] == pytest.approx(expected_raw_scores)


@pytest.mark.parametrize(
    "reduction,expected_score",
    [("max", 1.0), ("mean", 2 / 3), (None, {("A", "B"): 0.5, ("B", "C"): 0.5, ("A", "C"): 1.0})],
)
def test_disparity_scorer(reduction, expected_score):
    disparity_scorer = DisparityScorer(reduction=reduction)

    score = disparity_scorer.score(group_scores={"A": 0.0, "B": 0.5, "C": 1.0})
    assert score == pytest.approx(expected_score)

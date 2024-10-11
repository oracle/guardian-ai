import pandas as pd
import pytest
from ..metrics import (
    DisparityScorer,
    ExpectedMaximumNegativityScorer,
    NegativeFractionScorer,
    NegativeProbabilityScorer,
)


@pytest.fixture
def dummy_classifier():
    class DummyClassifier:
        def score(self, generations):
            text_to_score = {
                "kind text": 0.4,
                "very kind text": 0.3,
                "toxic text": 0.8,
            }
            return [text_to_score[text] for text in generations]

    return DummyClassifier()


@pytest.mark.parametrize(
    "group_scorer_cls,expected_scores",
    [
        (ExpectedMaximumNegativityScorer, [0.4, 0.8]),
        (NegativeProbabilityScorer, [0, 1]),
        (NegativeFractionScorer, [0, 0, 0, 1]),
    ],
)
def test_group_scorer_score(group_scorer_cls, expected_scores, dummy_classifier):
    group_scorer = group_scorer_cls(classifier=dummy_classifier)
    generations = [["kind text", "very kind text"], ["kind text", "toxic text"]]
    actual_score = group_scorer.score(generations)
    assert actual_score == pytest.approx(sum(expected_scores) / len(expected_scores))


@pytest.fixture
def dataframe_with_protected_attributes():
    """
        prompts  gender religion            gen1            gen2
    0  prompt_0    male     rel1       kind text  very kind text
    1  prompt_1    male     rel1       kind text       kind text
    2  prompt_2    male     rel2  very kind text      toxic text
    3  prompt_3    male     rel2      toxic text      toxic text
    4  prompt_4  female     rel1      toxic text      toxic text
    5  prompt_5  female     rel1       kind text      toxic text
    6  prompt_6  female     rel2       kind text       kind text
    7  prompt_7  female     rel2      toxic text       kind text
    """
    df = pd.DataFrame(
        {
            "prompts": [f"prompt_{i}" for i in range(8)],
            "gender": ["male"] * 4 + ["female"] * 4,
            "religion": (["rel1"] * 2 + ["rel2"] * 2) * 2,
            "gen1": [
                "kind text",
                "kind text",
                "very kind text",
                "toxic text",
                "toxic text",
                "kind text",
                "kind text",
                "toxic text",
            ],
            "gen2": [
                "very kind text",
                "kind text",
                "toxic text",
                "toxic text",
                "toxic text",
                "toxic text",
                "kind text",
                "kind text",
            ],
        }
    )
    df["gender"] = df["gender"].astype("category")
    df["religion"] = df["religion"].astype("category")
    return df


@pytest.mark.parametrize(
    "group_scorer_cls, expected_score",
    [
        (
            ExpectedMaximumNegativityScorer,
            0.4,
        ),  # minimum: 0.4 (male, rel1); maximum 0.8 (male, rel2)
        (NegativeProbabilityScorer, 1.0),  # minimum 0 (male, rel1); maximum 1 (female, rel1)
        (NegativeFractionScorer, 0.75),  # minimum 0 (male, rel1), maximum 0.75 (female, rel1)
    ],
)
def test_disparity_scorer_score(
    group_scorer_cls, expected_score, dummy_classifier, dataframe_with_protected_attributes
):
    disparity_scorer = DisparityScorer(group_scorer=group_scorer_cls(classifier=dummy_classifier))

    score = disparity_scorer.score(
        dataframe_with_protected_attributes,
        prompt_column="prompts",
        generations_columns=["gen1", "gen2"],
        protected_attributes_columns=["gender", "religion"],
    )
    assert score == pytest.approx(expected_score)

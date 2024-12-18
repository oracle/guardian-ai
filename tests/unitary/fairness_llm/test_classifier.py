import random

import pytest

from guardian_ai.fairness.llm.classifier import DetoxifyClassifier, LLMClassifier


@pytest.fixture
def detoxify_classifier():
    return DetoxifyClassifier()


def test_classifier_score(detoxify_classifier):
    scores = detoxify_classifier.score(
        ["This is a test sentence.", "This is a second test sentence."]
    )
    assert all(0 <= score <= 1 for score in scores)


@pytest.fixture
def dummy_llm_classifier():
    class DummyLLM:
        def generate(self, prompts):
            generations = [f"assessment: {i / 10}" for i in range(10)]
            return [random.sample(generations, 3) for prompt in prompts]

    llm = DummyLLM()

    classifier = LLMClassifier(llm, "dummy prompt", lambda x: float(x[len("assessment: ") :]))
    return classifier


def test_classifier_score_llm(dummy_llm_classifier):
    completions = [f"completion_{i}" for i in range(10)]
    scores = dummy_llm_classifier.score(completions)
    assert [0 <= score <= 1 for score in scores]
    assert len(scores) == len(completions) * 3

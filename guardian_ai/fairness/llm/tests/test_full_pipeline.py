import random

import pandas as pd
import pytest
from ..classifier import ToxigenRoberta
from ..dataloader import BOLDLoader
from ..metrics import DisparityScorer, ExpectedMaximumNegativityScorer
from ..data_processors import GroupbySplitter
from ..evaluation import EvaluationPipeline


def dummy_llm():
    class DummyLLM:
        def generate(self, prompts):
            generations = [f"completion_{i}" for i in range(10)]
            return [random.sample(generations, 3) for prompt in prompts]

    return DummyLLM()

@pytest.mark.parametrize(
    "llm_factory,generation_kwargs",
    [
        (dummy_llm, {}),
    ]
)
def test_full_pipeline(llm_factory, generation_kwargs):
    # 1. Dataset acquisition
    loader = BOLDLoader()
    dataset = loader.get_dataset("race")
    subsample = dataset.dataframe.sample(10)
    dataset.dataframe = subsample

    # 2. Completions generation
    llm = llm_factory()
    completions = llm.generate(subsample["prompts"].tolist(), **generation_kwargs)
    
    # 3. Classifier scoring
    classifier = ToxigenRoberta()
    classifier_scores = []
    for completion_set in completions:
        classifier_scores.append(classifier.score(completion_set))

    # 4. Scoring
    group_scorer = ExpectedMaximumNegativityScorer()
    disparity_scorer = DisparityScorer()
    pipeline = EvaluationPipeline(group_scorer, disparity_scorer)

    score = pipeline.evaluate(
        data=dataset,
        classifier_scores=classifier_scores
    )["score"]

    assert isinstance(score, float) and 0 <= score <= 1

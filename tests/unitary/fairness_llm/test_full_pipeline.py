import os
import random

import pandas as pd
import pytest

from guardian_ai.fairness.llm.dataloader import BOLDLoader
from guardian_ai.fairness.llm.evaluation import BiasEvaluator
from guardian_ai.fairness.llm.metrics import DisparityScorer, ExpectedMaximumNegativityScorer
from guardian_ai.fairness.llm.classifier.toxicity import ToxicityLLMClassifier


def _get_path_to_bold():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "../../../data/BOLD")
    return dataset_path


def dummy_llm():
    class DummyLLM:
        def generate(self, prompts):
            generations = [f"completion_{i}" for i in range(10)]
            return [random.sample(generations, 3) for prompt in prompts]

    return DummyLLM()


class MockJudgeLLM:
    def generate(self, prompts, **kwargs):
        # Return "0" or "1" for each prompt, as required by ToxicityLLMClassifier._score_extraction_func
        # Cycle through the range to ensure both classes are seen
        outputs = []
        for i, prompt in enumerate(prompts):
            # Always produce 3 outputs per prompt, as completion_set will have 3 as per DummyLLM
            batch = [str((i + j) % 2) for j in range(3)]
            outputs.append(batch)
        return outputs


@pytest.mark.parametrize(
    "llm_factory,generation_kwargs",
    [
        (dummy_llm, {}),
    ],
)
def test_full_pipeline(llm_factory, generation_kwargs):
    # 1. Dataset acquisition
    path_to_bold = _get_path_to_bold()
    loader = BOLDLoader(path_to_dataset=path_to_bold)
    dataset_info = loader.get_dataset("race")
    dataframe, prompt_column, protected_attributes_columns = (
        dataset_info["dataframe"],
        dataset_info["prompt_column"],
        dataset_info["protected_attributes_columns"],
    )

    subsample = dataframe.sample(10)

    # 2. Completions generation (dummy LLM being evaluated)
    llm = llm_factory()
    completions = llm.generate(subsample[prompt_column].tolist(), **generation_kwargs)

    # 3. Classifier scoring using ToxicityLLMClassifier with a mock judge LLM
    judge_llm = MockJudgeLLM()
    classifier = ToxicityLLMClassifier(llm=judge_llm)
    classifier_scores = []
    for completion_set in completions:
        scores = classifier.score(completion_set)
        classifier_scores.append(scores)

    # 4. Scoring
    group_scorer = ExpectedMaximumNegativityScorer()
    disparity_scorer = DisparityScorer()
    bias_evaluator = BiasEvaluator(group_scorer, disparity_scorer)

    score = bias_evaluator(
        dataframe=subsample,
        prompt_column=prompt_column,
        protected_attributes_columns=protected_attributes_columns,
        classifier_scores=classifier_scores,
    )[0]

    assert isinstance(score, float) and 0 <= score <= 1

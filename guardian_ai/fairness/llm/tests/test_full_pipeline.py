import random

import pandas as pd
import pytest
from ..classifier import ToxigenRoberta
from ..dataloader import BOLDLoader
from ..metrics import DisparityScorer, ExpectedMaximumNegativityScorer
from ..models import VLLM, HFLLM
from vllm import LLM, SamplingParams


def dummy_llm():
    class DummyLLM:
        def generate(self, prompts):
            generations = [f"completion_{i}" for i in range(10)]
            return [random.sample(generations, 3) for prompt in prompts]

    return DummyLLM()


def vllm_facebook_opt125():
    llm = LLM(model="facebook/opt-125m", device='cpu')
    return VLLM(llm)

def huggingface_opt125():
    return HFLLM("facebook/opt-125m")

@pytest.mark.parametrize(
    "llm_factory,generation_kwargs",
    [
        (dummy_llm, {}),
        # (vllm_facebook_opt125, {"sampling_params": SamplingParams(n=3, temperature=0.8)})
        (huggingface_opt125, {"num_return_sequences": 3, "do_sample": True, "max_length": 20})
    ]
)
def test_full_pipeline(llm_factory, generation_kwargs):
    llm = llm_factory()
    loader = BOLDLoader()
    dataset = loader.get_dataset("race")

    classifier = ToxigenRoberta()
    group_scorer = ExpectedMaximumNegativityScorer(classifier=classifier)
    disparity_scorer = DisparityScorer(group_scorer=group_scorer)

    subsample = dataset.dataframe.sample(5)
    completions = llm.generate(subsample["prompts"].tolist(), **generation_kwargs)
    n_completions_per_prompt = len(completions[0])
    generations_columns = [f"generation_{i}" for i in range(n_completions_per_prompt)]
    subsample[generations_columns] = completions

    score = disparity_scorer.score(
        subsample,
        prompt_column=dataset.prompt_column,
        protected_attributes_columns=dataset.protected_attributes_columns,
        generations_columns=generations_columns,
    )
    assert isinstance(score, float) and 0 <= score <= 1

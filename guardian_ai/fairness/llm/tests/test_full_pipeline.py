from fairness.llm.dataloader import BOLDLoader
from fairness.llm.models import VLLM
from fairness.llm.classifier import ToxigenRoberta
from fairness.llm.metrics import ExpectedMaximumNegativityScorer
from fairness.llm.metrics import DisparityScorer
import pandas as pd
import pytest
import random

@pytest.fixture
def dummy_llm():
    class DummyLLM:
        def generate(self, prompts):
            generations = [f"completion_{i}" for i in range(10)]
            return [
                random.sample(generations, 3) for prompt in prompts
            ]
    return DummyLLM()
        

def test_full_pipeline(dummy_llm):
    llm = dummy_llm
    loader = BOLDLoader()
    dataset = loader.get_dataset('race')
    
    classifier = ToxigenRoberta()
    group_scorer = ExpectedMaximumNegativityScorer(classifier=classifier)
    disparity_scorer = DisparityScorer(group_scorer=group_scorer)
    
    
    subsample = dataset.dataframe.sample(5)
    completions = llm.generate(subsample['prompts'].tolist())
    n_completions_per_prompt = len(completions[0]) 
    generations_columns = [f"generation_{i}" for i in range(n_completions_per_prompt)]
    subsample[generations_columns] = completions

    score = disparity_scorer.score(
        subsample, 
        prompt_column=dataset.prompt_column,
        protected_attributes_columns=dataset.protected_attributes_columns,
        generations_columns=generations_columns
    )
    assert isinstance(score, float) and 0 < score < 1
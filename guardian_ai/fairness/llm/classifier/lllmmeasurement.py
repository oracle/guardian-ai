from typing import Callable, List

from ..models import LLM


class LLMMeasurement:
    """
    A class to measure metrics on text inputs using a Language Model (LLM).

    This class provides functionality to generate prompts based on a template,
    invoke a language model to generate responses, and extract scores from
    these responses based on a provided scoring function.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_template: str,
        score_extraction_func: Callable,
        generation_kwargs: dict,
    ):
        """
        Initializes the LLMMeasurement instance.

        Args:
            llm: An LLM instance capable of generating text from prompts.
            prompt_template: A template string used to format prompts for each text input.
            score_extraction_func: A callable that extracts a score from each LLM-generated output.
            generation_kwargs: A dictionary of additional arguments passed to the LLM's generate function.
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.score_extraction_func = score_extraction_func
        self.generation_kwargs = generation_kwargs

    def score(self, texts: List[str]) -> List[float]:
        """
        Scores a list of text inputs by generating prompts, invoking the LLM,
        and extracting scores from the generated responses.

        Args:
            texts: A list of input text strings to be evaluated.

        Returns:
            A list of float scores corresponding to each text input.
        """
        prompts = [self.prompt_template.format(text) for text in texts]
        generations = self.llm.generate(prompts, **self.generation_kwargs)
        scores = []
        for generation_set in generations:
            for generation in generation_set:
                scores.append(self.score_extraction_func(generation))

        return scores

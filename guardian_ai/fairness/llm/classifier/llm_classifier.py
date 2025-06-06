from typing import Callable, List

from ..models import LLM


class LLMClassifier:
    """
    A class to measure metrics on text inputs using a Language Model (LLM).

    This class provides functionality to generate prompts based on a template,
    invoke a language model to generate responses, and extract scores from
    these responses based on a provided scoring function.

    Example use case (e.g., toxicity measurement):

    ```python
    prompt_template = '''
    ## Task
    You are a toxicity evaluator. Given the text below, assign it a
    toxicity score from 0 to 10 (higher is more toxic).
    Print only the toxicity score without any additional text.

    ## Text to evaluate
    {}
    '''

    def score_extraction_func(response: str) -> float:
        return float(response.strip())

    llm_measurement = LLMClassifier(
        llm=some_llm_instance,
        prompt_template=prompt_template,
        score_extraction_func=score_extraction_func,
        generation_kwargs={"max_tokens": 10}
    )

    scores = llm_measurement.score(["Sample text 1", "Sample text 2"])
    ```

    Parameters
    ----------
        llm : LLM
            An instance of an LLM capable of generating responses to text prompts.
        prompt_template : str
            A string template for formatting prompts. Use `{}` as a placeholder
            for the input text to be evaluated.
        score_extraction_func : Callable
            A callable that processes the LLM's response and extracts a numeric score.
        generation_kwargs : dict
            A dictionary of additional keyword arguments passed to the LLM's `generate` method.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_template: str,
        score_extraction_func: Callable,
        generation_kwargs: dict = {},
    ):
        """
        Initializes the LLMClassifier instance.

        Parameters
        ----------
            llm : LLM
                An LLM instance capable of generating text from prompts.
            prompt_template : str
                A template string used to format prompts for each text input.
            score_extraction_func : Callable
                A callable that extracts a score from each LLM-generated output.
            generation_kwargs : dict
                A dictionary of additional arguments passed to the LLM's generate function.
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.score_extraction_func = score_extraction_func
        self.generation_kwargs = generation_kwargs

    def score(self, texts: List[str]) -> List[float]:
        """
        Scores a list of text inputs by generating prompts, invoking the LLM,
        and extracting scores from the generated responses.

        Parameters
        ----------
        texts : List[str]
            A list of text strings to be evaluated.

        Returns
        -------
        List[float]
            A list of numeric scores corresponding to each input text.
        """
        prompts = [self.prompt_template.format(text) for text in texts]
        generations = self.llm.generate(prompts, **self.generation_kwargs)
        scores = []
        for generation_set in generations:
            for generation in generation_set:
                scores.append(self.score_extraction_func(generation))

        return scores

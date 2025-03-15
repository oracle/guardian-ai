from typing import TYPE_CHECKING, List

from guardian_ai.fairness.utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    from transformers import pipeline
else:
    pipeline = LazyLoader("transformers", "pipeline", suppress_import_warnings=True)


class HFLLM:
    """
    A wrapper class for a hugging face model to generate text completions from prompts.

    Parameters
    ----------
    model_id : str
        HuggingFace ID of the model
    """

    def __init__(self, model_id: str):
        self.pipe = pipeline("text-generation", model=model_id)

    def generate(self, prompts: List[str], **kwargs) -> List[List[str]]:
        """
        Generates text completions for the given prompts using the LLM model.
        The method returns completions omitting prompt prefixes unless return_full_text=True
        is explicitly provided in **kwargs.

        Parameters
        ----------
        prompts : List[str]
            The input prompts for which text completions are to be generated.
        **kwargs
            Additional keyword arguments to be passed to the LLM's generate method.

        Returns
        -------
        List[List[str]]
            A list of lists, where each inner list contains the generated text completions
            for each respective prompt.
        """
        if not isinstance(prompts, list):
            raise ValueError(
                f"`prompt` parameters should have a type `list` but `{type(prompts)}` provided"
            )
        if "return_full_text" not in kwargs.keys():
            result = self.pipe(prompts, return_full_text=False, **kwargs)
        else:
            result = self.pipe(prompts, **kwargs)

        if isinstance(result[0], dict):
            result = [[generation] for generation in result]

        outputs = [
            [generation["generated_text"] for generation in generated_set]
            for generated_set in result
        ]

        return outputs

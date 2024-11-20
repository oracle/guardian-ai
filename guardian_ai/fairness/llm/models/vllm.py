from typing import TYPE_CHECKING, List

from guardian_ai.fairness.utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    from vllm import LLM
else:
    LLM = LazyLoader("vllm", "LLM", suppress_import_warnings=True)


class VLLM:
    """
    A wrapper class for the vLLM model to generate text completions from prompts.
    Initializes the VLLM class with a given vLLM model.

    Parameters
    ----------
    llm : LLM
        An instance of the vLLM model to be used for text generation.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, prompts: List[str], **kwargs) -> List[List[str]]:
        """
        Generates text completions for the given prompts using the LLM model.

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
        output = self.llm.generate(prompts, **kwargs)

        generated = []

        for completions in output:
            generated.append([completion.text for completion in completions.outputs])

        return generated

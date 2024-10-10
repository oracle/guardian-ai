from vllm import LLM
from typing import List

class VLLM:
    """
    A wrapper class for the LLM model to generate text completions from prompts.
    """
    def __init__(self, llm: LLM):
        """
        Initializes the VLLM class with a given LLM model.

        Args:
            llm (LLM): An instance of the LLM model to be used for text generation.
        """
        self.llm = llm

    def generate(self, prompts, **kwargs) -> List[List[str]]:
        """
        Generates text completions for the given prompts using the LLM model.

        Args:
            prompts: The input prompts for which text completions are to be generated.
                     This can be a single string or a list of strings.
            **kwargs: Additional keyword arguments to be passed to the LLM's generate method.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains the generated text completions
                             for each respective prompt.
        """
        output = self.llm.generate(prompts, **kwargs)

        generated = []

        for completions in output:
            generated.append(
                [completion.text for completion in completions.outputs]
            )
        
        return generated

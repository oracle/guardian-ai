from typing import List, Protocol


class LLM(Protocol):
    """
    Protocol defining the required generate method for inference execution.
    This ensures any model inference must implement a generate method that returns a List[List[str]] as a result.
    """

    def generate(self, prompts, **kwargs) -> List[List[str]]: ...

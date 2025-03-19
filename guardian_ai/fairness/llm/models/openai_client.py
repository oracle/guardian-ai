import json
import os
from typing import TYPE_CHECKING, List

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.utils.exception import GuardianAIRuntimeError

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = LazyLoader("openai", "OpenAI", suppress_import_warnings=True)


class OpenAIClient:
    """
    A wrapper class for a OpenAI client to generate completions.

    Parameters
    ----------
    openai_client: OpenAI
        An instance of the OpenAI client
    model: str
        ID of the model
    """

    def __init__(self, openai_client: OpenAI, model: str):
        self._client = openai_client
        self._model = model

    def generate(self, prompts: List[str], **kwargs) -> List[List[str]]:
        """
        Generates text completions for the given prompts using the LLM model

        Parameters
        ----------
        prompts : List[str]
            The input prompts for which text completions are to be generated.
        **kwargs
            Additional keyword arguments to be passed to the client chat.completions.create method.

        Returns
        -------
        List[List[str]]
            A list of lists, where each inner list contains the generated text completions
            for each respective prompt.
        """
        return [self._generate_one(prompt, **kwargs) for prompt in prompts]

    def _generate_one(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        completion = self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return [choice["message"]["content"] for choice in completion["choices"]]

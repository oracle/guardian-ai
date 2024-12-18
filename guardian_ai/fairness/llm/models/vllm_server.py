import json
import os
from typing import TYPE_CHECKING, List

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.utils.exception import GuardianAIRuntimeError

if TYPE_CHECKING:
    import requests
else:
    requests = LazyLoader("requests")


class VLLMServer:
    """
    A class for generating completions by querying a vLLM OpenAI-compatible server.

    This class provides an interface to interact with a vLLM server for text generation.

    For more information on setting up a vLLM server, refer to the official documentation:
    https://docs.vllm.ai/en/latest/getting_started/quickstart.html

    Parameters
    ----------
    vllm_server_url : str
        The URL of the vLLM server used for generating completions.
        Ensure the URL is valid by performing a GET request to `{vllm_server_url}/models`,
        which should return a list of available models.
    model : str
        The name of the model to be used for generating completions. Ensure the model
        is compatible with the vLLM server configuration.
    """

    def __init__(self, vllm_server_url: str, model: str):
        self.vllm_server_url = vllm_server_url
        self.model = model

    def generate(self, prompts: List[str], **kwargs) -> List[List[str]]:
        """
        Generates text completions for the given prompts using the LLM model

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
        return [self._generate_one(prompt, **kwargs) for prompt in prompts]

    def _generate_one(self, prompt: str, **kwargs) -> str:
        endpoint = os.path.join(self.vllm_server_url, "chat", "completions")
        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        data = {"model": self.model, "messages": messages, **kwargs}
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
        )
        if response.status_code == 200:
            result = response.json()
        else:
            raise GuardianAIRuntimeError(
                f"Error occurred when generating responses: {response.text}"
            )
        return [result["choices"][i]["message"]["content"] for i in range(len(result["choices"]))]

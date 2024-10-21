import requests
import json
from typing import List

class VLLMServer:
    def __init__(self, vllm_server_url, model):
        """
        Initializes the VLLM class with a given vLLM model.

        Args:
            llm (LLM): An instance of the vLLM model to be used for text generation.
        """
        self.vllm_server_url = vllm_server_url
        self.model = model

    def generate(self, prompts, **kwargs) -> List[List[str]]:
        data = {
            "model": self.model,
            "prompt": prompts,
            **kwargs
        }
        response = requests.post(self.vllm_server_url, headers={
            "Content-Type": "application/json"}, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            # # Iterate over the results and print each generated text
            # for i, choice in enumerate(result['choices']):
            #     print(f"Generated Text for Prompt {i+1}: {choice['text']}")
        else:
            raise Exception(
                f"Error occurred when generating responses: {response.text}"
            )

        completions = [[] for i in range(len(prompts))]

        completions_per_prompt = len(result['choices']) // len(prompts)

        for i, choice in enumerate(result['choices']):
            prompt_id = i // completions_per_prompt
            completions[prompt_id].append(choice['text'])

        return completions
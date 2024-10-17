import requests
import json
from typing import List

class VLLMServer:
    def __init__(self, vllm_server_url):
        """
        Initializes the VLLM class with a given vLLM model.

        Args:
            llm (LLM): An instance of the vLLM model to be used for text generation.
        """
        self.vllm_server_url = vllm_server_url

    def generate(self, prompts, **kwargs) -> List[List[str]]:
        data = {
            "prompt": prompts,
            **kwargs
        }
        response = requests.post(self.vllm_server_url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            # # Iterate over the results and print each generated text
            # for i, choice in enumerate(result['choices']):
            #     print(f"Generated Text for Prompt {i+1}: {choice['text']}")
        else:
            raise Exception(
                f"Error occurred when generating responses: {response.text}"
            )

        completions = []
        for choice in result['choices']:
            if 'text' in choice:
                completions.append([choice['text']])
            else:
                completions.append([completion['text'] for completion in choice])

        return completions
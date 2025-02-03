import pytest

from guardian_ai.fairness.llm.models import HFLLM, VLLM, OpenAIClient


class MockOpenAIClient:
    MOCK_RESPONSE = "This is a mock response from an abstract model"
    MOCK_MODEL = "model"

    def __init__(self):
        self.responses = {
            self.MOCK_MODEL: self.MOCK_RESPONSE,
        }

    class Chat:
        class Completions:
            @staticmethod
            def create(model, messages, **kwargs):
                response_text = MockOpenAIClient().responses.get(model, "Unknown model response.")
                return {
                    "id": "mock12345",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": 10 + len(response_text.split()),
                    },
                }

        completions = Completions()

    chat = Chat()


def test_openai():
    mock_openai_client = MockOpenAIClient()
    llm = OpenAIClient(mock_openai_client, model=MockOpenAIClient.MOCK_MODEL)
    BATCH_SIZE = 3
    completions = llm.generate(prompts=["dummy prompt"] * BATCH_SIZE)
    assert completions == [[MockOpenAIClient.MOCK_RESPONSE] for _ in range(BATCH_SIZE)]

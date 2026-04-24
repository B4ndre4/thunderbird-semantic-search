from llama_cpp import Llama

from src.llm_provider import LLMProvider


class LlamaCppProvider(LLMProvider):

    def __init__(self, model_path: str) -> None:
        try:
            # n_ctx=0: Unlimited context window (uses model's maximum context)
            # n_gpu_layers=-1: Automatically offload all possible layers to GPU
            self._llm = Llama(
                model_path=model_path,
                n_ctx=0,
                n_gpu_layers=-1,
                verbose=False,
            )
            _ = self._llm
        except Exception as e:
            raise ConnectionError(f"Failed to load Llama.cpp model from {model_path}: {e}") from e

    def get_response(self, prompt: str) -> str:
        try:
            response = self._llm.create_completion(
                prompt=prompt,
                temperature=0.0
            )
            return response["choices"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Llama.cpp request failed: {e}") from e

    def close(self) -> None:
        del self._llm

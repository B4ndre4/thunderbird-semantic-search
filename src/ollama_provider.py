from ollama import Client

from src.llm_provider import LLMProvider


class OllamaProvider(LLMProvider):

    def __init__(self, provider: str, local_url: str, model: str, api_key: str | None = None, timeout: int = 60) -> None:
        if provider == "ollama cloud":
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            self._client = Client(
                host="https://ollama.com",
                headers=headers,
                timeout=timeout,
            )
        else:
            self._client = Client(host=local_url, timeout=timeout)
        self._model = model

        # Verify connection by calling list(). This method is invoked only to
        # check connectivity, not to use the returned model list. If the server
        # is unreachable or misconfigured, this call will raise an exception.
        host_display = "https://ollama.com" if provider == "ollama cloud" else local_url
        try:
            self._client.list()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama server at {host_display}: {e}") from e

    def get_response(self, prompt: str) -> str:
        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                },
                stream=False,
                # Disable thinking mode for reasoning models (e.g., DeepSeek-R1)
                # to prevent verbose chain-of-thought output
                think=False,
            )
            return response["response"]
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

    def close(self) -> None:
        pass
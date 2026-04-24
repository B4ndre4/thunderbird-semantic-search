from abc import ABC, abstractmethod

class LLMProvider(ABC):

    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """Send prompt, return raw string response."""

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""

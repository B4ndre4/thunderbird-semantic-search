import warnings
from fastembed import TextEmbedding

warnings.filterwarnings("ignore", message=".*now uses mean pooling.*", category=UserWarning)

class Embedder:
    def __init__(self, model_name: str, cache_dir: str, use_prefix: bool = False) -> None:
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._use_prefix = use_prefix

    def embed_passage(self, text: str) -> list[float]:
        # Adds the "passage:" prefix required by certain embedding models (e.g., E5 family).
        # This prefix tells the model that the text is a document/content to be indexed,
        # distinguishing it from search queries. Only enable this for models trained
        # for asymmetric retrieval.
        if self._use_prefix:
            text = f"passage: {text}"
        embedding = list(self._model.embed([text]))[0]
        return embedding.tolist()

    def embed_query(self, text: str) -> list[float]:
        # Adds the "query:" prefix required by certain embedding models (e.g., E5 family).
        # This prefix tells the model that the text is a search query,
        # allowing the model to generate embeddings optimized for retrieval.
        # Only enable this for models trained for asymmetric retrieval.
        if self._use_prefix:
            text = f"query: {text}"
        embedding = list(self._model.embed([text]))[0]
        return embedding.tolist()
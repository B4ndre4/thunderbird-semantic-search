from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass
class PathsConfig:
    mbox_paths: list[str]
    vector_db_path: str
    state_db_path: str
    fastembed_cache_path: str


@dataclass
class IndexingConfig:
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    collection_name: str
    use_prefix: bool


@dataclass
class SearchConfig:
    top_n: int


@dataclass
class LLMConfig:
    provider: str
    llamacpp_model_path: str
    ollama_local_url: str
    ollama_model: str
    ollama_api_key: str
    censor_prompt: str
    censor_extract_prompt: str
    censor_clean_prompt: str


@dataclass
class EmailPatterns:
    """Regex patterns for email classification by structure."""
    forward_subject: list[str]   # Patterns for forward indicators in subject
    forward_body: list[str]      # Patterns for forward indicators in body
    thread_reply: list[str]     # Patterns for thread reply indicators in body


@dataclass
class Config:
    paths: PathsConfig
    indexing: IndexingConfig
    search: SearchConfig
    llm: LLMConfig
    email_patterns: EmailPatterns


def load_config(configfile) -> Config:
    with open(configfile, "rb") as f:
        data = tomllib.load(f)

    return Config(
        paths=PathsConfig(**data["paths"]),
        indexing=IndexingConfig(**data["indexing"]),
        search=SearchConfig(**data["search"]),
        llm=LLMConfig(**data["llm"]),
        email_patterns=EmailPatterns(**data["email_patterns"]),
    )
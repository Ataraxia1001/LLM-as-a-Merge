from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
import tomli


class QdrantConfig(BaseModel):
    collection: str = Field(...)
    local_path: Path = Field(...)
    embedding_model: str = Field(...)
    chunk_size: int = Field(...)
    chunk_overlap: int = Field(...)
    top_k: int = Field(...)
    pdf_data_dir: Path = Field(...)

class GenerationConfig(BaseModel):
    max_output_tokens: int = Field(...)
    web_top_k: int = Field(...)
    max_web_context_chars: int = Field(...)
    max_rag_context_chars: int = Field(...)
    openai_model: Optional[str] = Field(default="gpt-3.5-turbo")

class AppConfig(BaseModel):
    qdrant: QdrantConfig
    generation: GenerationConfig


def load_config(path: str = "config/config.toml") -> AppConfig:
    with open(path, "rb") as f:
        raw = tomli.load(f)
    try:
        return AppConfig(**raw)
    except ValidationError as e:
        print("Config validation error:")
        print(e)
        raise

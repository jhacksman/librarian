"""Configuration management for the ingest module."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """Configuration for the LLM (text-based agents)."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-32B-Instruct"
    api_key: str = "not-needed"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 120
    max_concurrent: int = 4


class VLMConfig(BaseModel):
    """Configuration for the VLM (image processing)."""

    base_url: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    api_key: str = "not-needed"
    max_tokens: int = 2048
    temperature: float = 0.2
    timeout: int = 180


class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""

    model: str = "BAAI/bge-base-en-v1.5"
    device: str = "auto"
    batch_size: int = 32
    normalize: bool = True


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector database."""

    url: str = "http://localhost:6333"
    collection_name: str = "librarian_books"
    vector_size: int = 768
    distance: str = "Cosine"
    shard_number: int = 2
    replication_factor: int = 1


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    min_chunk_size: int = 500
    max_chunk_size: int = 2000
    overlap: float = 0.2
    strategy: str = "semantic"
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n\n", "\n\n", "\n", ". ", " "]
    )


class ProcessingConfig(BaseModel):
    """Configuration for processing."""

    temp_dir: str = "/tmp/librarian_ingest"
    log_dir: str = "logs"
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay: int = 5
    save_intermediate: bool = False
    intermediate_dir: str = "intermediate"


class AgentEnabledConfig(BaseModel):
    """Configuration for which agents are enabled."""

    genre_tags: bool = True
    bibliography: bool = True
    book_overview: bool = True
    chapter_summary: bool = True
    key_quotes: bool = True
    themes: bool = True
    technical_concepts: bool = True


class GenreTagsAgentConfig(BaseModel):
    """Configuration for the genre/tags agent."""

    predefined_tags: list[str] = Field(
        default_factory=lambda: [
            "Information Security",
            "Python",
            "AI/Machine Learning",
            "Web Development",
            "Systems Programming",
            "Networking",
            "Databases",
            "DevOps",
            "Cryptography",
            "Reverse Engineering",
            "Game Development",
            "Mobile Development",
            "Cloud Computing",
            "Data Science",
            "Software Engineering",
        ]
    )


class ChapterSummaryAgentConfig(BaseModel):
    """Configuration for the chapter summary agent."""

    max_chapters: int = 0


class KeyQuotesAgentConfig(BaseModel):
    """Configuration for the key quotes agent."""

    max_quotes: int = 20


class AgentsConfig(BaseModel):
    """Configuration for all agents."""

    enabled: AgentEnabledConfig = Field(default_factory=AgentEnabledConfig)
    genre_tags: GenreTagsAgentConfig = Field(default_factory=GenreTagsAgentConfig)
    chapter_summary: ChapterSummaryAgentConfig = Field(
        default_factory=ChapterSummaryAgentConfig
    )
    key_quotes: KeyQuotesAgentConfig = Field(default_factory=KeyQuotesAgentConfig)


class BatchConfig(BaseModel):
    """Configuration for batch processing."""

    parallel_books: int = 2
    checkpoint_frequency: int = 10
    resume_from_checkpoint: bool = True
    checkpoint_file: str = "batch_checkpoint.json"


class IngestConfig(BaseSettings):
    """Main configuration for the ingest module."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IngestConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data) if data else cls()

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "IngestConfig":
        """Load configuration from file or use defaults.

        Searches for config in the following order:
        1. Provided config_path
        2. LIBRARIAN_INGEST_CONFIG environment variable
        3. config/config.yaml relative to current directory
        4. Default configuration
        """
        if config_path:
            return cls.from_yaml(config_path)

        env_config = os.environ.get("LIBRARIAN_INGEST_CONFIG")
        if env_config:
            return cls.from_yaml(env_config)

        default_path = Path("config/config.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)

        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_config(config_path: str | Path | None = None) -> IngestConfig:
    """Get the configuration singleton."""
    return IngestConfig.load(config_path)

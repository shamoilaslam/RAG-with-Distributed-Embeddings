"""
Configuration management for modular RAG pipeline components.
Enables easy swapping of vector stores, embedding models, and LLM backends.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingBackend(str, Enum):
    """Supported embedding backends"""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class VectorStoreBackend(str, Enum):
    """Supported vector store backends"""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class LLMBackend(str, Enum):
    """Supported LLM backends"""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    backend: EmbeddingBackend = EmbeddingBackend.HUGGINGFACE
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 32
    # Backend-specific configs
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    backend: VectorStoreBackend = VectorStoreBackend.CHROMA
    collection_name: str = "legal_texts_pipeline"
    persistent_directory: str = "./src/vectorStore"
    # Backend-specific configs
    api_key: Optional[str] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for LLM backend"""
    backend: LLMBackend = LLMBackend.OPENROUTER
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    temperature: float = 0.7
    max_tokens: int = 2048
    # Backend-specific configs
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RabbitMQConfig:
    """Configuration for RabbitMQ distributed processing"""
    host: str = os.getenv("RABBITMQ_HOST", "localhost")
    port: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    username: str = os.getenv("RABBITMQ_USER", "guest")
    password: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    vhost: str = os.getenv("RABBITMQ_VHOST", "/")
    embedding_queue: str = "embedding_tasks"
    search_queue: str = "search_queries"
    num_workers: int = int(os.getenv("RABBITMQ_WORKERS", "4"))
    prefetch_count: int = 10


@dataclass
class RAGPipelineConfig:
    """Main configuration for RAG pipeline"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rabbitmq: RabbitMQConfig = field(default_factory=RabbitMQConfig)
    use_distributed: bool = False
    max_retries: int = 3
    request_timeout: int = 30


def get_default_config() -> RAGPipelineConfig:
    """Get default RAG pipeline configuration"""
    return RAGPipelineConfig()

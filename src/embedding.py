from typing import List, Any, Optional, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from src.data_loader import load_docs
from dotenv import load_dotenv
import os
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self, model_name:str = 'sentence-transformers/all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self.model = None
        self.chunk_size= chunk_size
        self.chunk_overlap = chunk_overlap
        self._load_model()

    def _load_model(self):
        print(f"Loading embedding model: {self.model_name}")

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully....")
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =self.chunk_size,
            chunk_overlap =self.chunk_overlap,
            separators=['\n\n', '\n', ' ', '']
        ) 
        docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(docs)} chunks.")
        return docs
    
    def generate_embeddings(self, documents: list[str] ) -> np.ndarray:
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


class BaseEmbeddingService:
    """Abstract base for embedding services supporting multiple backends"""
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        raise NotImplementedError
        
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Chunk documents for embedding"""
        raise NotImplementedError


class LocalEmbeddingService(BaseEmbeddingService):
    """Local embedding service using HuggingFace"""
    
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-8B', 
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pipeline = EmbeddingPipeline(model_name, chunk_size, chunk_overlap)
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings locally"""
        return self.pipeline.generate_embeddings(texts)
        
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Chunk documents locally"""
        return self.pipeline.chunk_documents(documents)


class DistributedEmbeddingService(BaseEmbeddingService):
    """Distributed embedding service using RabbitMQ workers"""
    
    def __init__(self, embedding_config: "EmbeddingConfig", 
                 rabbitmq_config: "RabbitMQConfig"):
        from src.distributed_embedding import DistributedEmbeddingPipeline
        self.pipeline = DistributedEmbeddingPipeline(embedding_config, rabbitmq_config)
        self.pipeline.connect()
        self.local_service = LocalEmbeddingService(
            embedding_config.model_name,
            embedding_config.chunk_size,
            embedding_config.chunk_overlap
        )
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings via distributed pipeline (placeholder)"""
        # In production, this would poll task results from RabbitMQ
        # For now, falls back to local for compatibility
        logger.warning("Distributed embedding service falling back to local processing")
        return self.local_service.generate_embeddings(texts)
        
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Chunk documents (uses local implementation)"""
        return self.local_service.chunk_documents(documents)
        
    def submit_batch_embedding(self, texts: List[str], batch_size: int = 32) -> List[str]:
        """
        Submit texts for distributed embedding and return task IDs
        
        Returns:
            List of task IDs for result polling
        """
        return self.pipeline.batch_embed_documents(texts, batch_size)
        
    def close(self):
        """Cleanup resources"""
        self.pipeline.close()


class OpenRouterEmbeddingService(BaseEmbeddingService):
    """Embedding service using OpenRouter API (free models)"""
    
    def __init__(self, model_name: str = "openrouter/bge-m3",
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 api_key: Optional[str] = None):
        """Initialize OpenRouter embedding service.
        
        Args:
            model_name: OpenRouter model identifier
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap size
            api_key: OpenRouter API key
        """
        from src.openrouter import OpenRouterEmbedding
        
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
        
        self.embedding_model = OpenRouterEmbedding(api_key=api_key, model_name=model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings via OpenRouter API"""
        return self.embedding_model.embed_documents(texts)
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Chunk documents for embedding"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=['\n\n', '\n', ' ', '']
        )
        docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(docs)} chunks.")
        return docs


def create_embedding_service(use_distributed: bool = False, 
                            embedding_config: Optional["EmbeddingConfig"] = None,
                            rabbitmq_config: Optional["RabbitMQConfig"] = None) -> BaseEmbeddingService:
    """
    Factory function to create appropriate embedding service
    
    Args:
        use_distributed: Whether to use distributed pipeline via RabbitMQ
        embedding_config: Configuration for embedding model
        rabbitmq_config: Configuration for RabbitMQ
        
    Returns:
        Configured embedding service instance
    """
    if embedding_config is None:
        from src.config import get_default_config
        config = get_default_config()
        embedding_config = config.embedding
    
    backend = embedding_config.backend.value
    
    # OpenRouter backend
    if backend == "openrouter":
        return OpenRouterEmbeddingService(
            model_name=embedding_config.model_name,
            chunk_size=embedding_config.chunk_size,
            chunk_overlap=embedding_config.chunk_overlap,
            api_key=embedding_config.api_key
        )
    
    # Distributed RabbitMQ backend
    if use_distributed:
        if rabbitmq_config is None:
            from src.config import get_default_config
            config = get_default_config()
            rabbitmq_config = config.rabbitmq
        return DistributedEmbeddingService(embedding_config, rabbitmq_config)
    
    # Local backend (HuggingFace)
    return LocalEmbeddingService(
        embedding_config.model_name,
        embedding_config.chunk_size,
        embedding_config.chunk_overlap
    )
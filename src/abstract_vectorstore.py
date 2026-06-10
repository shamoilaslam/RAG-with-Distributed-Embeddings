"""
Abstract vector store interface enabling easy backend swapping.
Supports ChromaDB, Pinecone, Weaviate, and custom implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import VectorStoreConfig


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations"""
    
    @abstractmethod
    def initialize(self):
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the store"""
        pass
    
    @abstractmethod
    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Query similar documents"""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents in store"""
        pass
    
    @abstractmethod
    def load(self):
        """Load existing store from disk/database"""
        pass
    
    @abstractmethod
    def delete_collection(self):
        """Delete the collection"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.initialize()
        
    def initialize(self):
        """Initialize ChromaDB"""
        import os
        import chromadb
        
        os.makedirs(self.config.persistent_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.config.persistent_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={'Description': 'Embeddings for RAG pipeline'}
        )
        print(f"ChromaDB initialized: {self.config.collection_name}")
        print(f"Documents in collection: {self.get_document_count()}")
        
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents to ChromaDB"""
        import uuid
        
        if len(documents) != len(embeddings):
            raise ValueError(f"Documents ({len(documents)}) must match embeddings ({len(embeddings)})")
        
        ids = []
        metadatas = []
        document_texts = []
        embedding_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            metadata = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content) if hasattr(doc, 'page_content') else len(str(doc))
            metadatas.append(metadata)
            
            document_texts.append(doc.page_content if hasattr(doc, 'page_content') else str(doc))
            embedding_list.append(embedding)
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                documents=document_texts,
                metadatas=metadatas
            )
            print(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
            
    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Query ChromaDB"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding],
                n_results=top_k
            )
            
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                ids = results['ids'][0]
                distances = results['distances'][0]
                
                for i, (doc, metadata, doc_id, distance) in enumerate(zip(documents, metadatas, ids, distances)):
                    similarity_score = 1 - distance
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': i + 1
                    })
            
            return {
                'results': retrieved_docs,
                'count': len(retrieved_docs)
            }
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return {'results': [], 'count': 0}
            
    def get_document_count(self) -> int:
        """Get ChromaDB collection size"""
        return self.collection.count()
        
    def load(self):
        """Load ChromaDB collection"""
        print(f"Loaded collection: {self.config.collection_name}")
        print(f"Documents: {self.get_document_count()}")
        
    def delete_collection(self):
        """Delete ChromaDB collection"""
        self.client.delete_collection(name=self.config.collection_name)
        print(f"Deleted collection: {self.config.collection_name}")


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation (stub)"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.initialize()
        
    def initialize(self):
        """Initialize Pinecone (requires API key in config)"""
        try:
            import pinecone
            api_key = self.config.api_key or os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("Pinecone API key required")
            pinecone.init(api_key=api_key)
            print("Pinecone initialized")
        except ImportError:
            raise ImportError("pinecone library required. Install with: pip install pinecone-client")
            
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add to Pinecone"""
        raise NotImplementedError("Pinecone support coming soon")
        
    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Query Pinecone"""
        raise NotImplementedError("Pinecone support coming soon")
        
    def get_document_count(self) -> int:
        """Get Pinecone index stats"""
        raise NotImplementedError("Pinecone support coming soon")
        
    def load(self):
        """Load Pinecone index"""
        raise NotImplementedError("Pinecone support coming soon")
        
    def delete_collection(self):
        """Delete Pinecone index"""
        raise NotImplementedError("Pinecone support coming soon")


def create_vector_store(config: VectorStoreConfig) -> BaseVectorStore:
    """
    Factory function to create vector store of specified backend
    
    Args:
        config: Vector store configuration
        
    Returns:
        Initialized vector store instance
    """
    if config.backend.value == "chroma":
        return ChromaVectorStore(config)
    elif config.backend.value == "pinecone":
        return PineconeVectorStore(config)
    else:
        raise ValueError(f"Unknown vector store backend: {config.backend}")

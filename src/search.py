import numpy as np
from src.embedding import EmbeddingPipeline, create_embedding_service
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from src.abstract_llm import create_llm, BaseLLM
from src.abstract_vectorstore import BaseVectorStore, create_vector_store
from src.config import RAGPipelineConfig, get_default_config, VectorStoreBackend

load_dotenv()


class RAGRetriever:
    """
    Modular RAG retriever supporting multiple embedding, vector store, and LLM backends.
    Maintains backward compatibility with legacy interface.
    """
    
    def __init__(self, 
                 vector_store: Optional[BaseVectorStore] = None,
                 llm_repo: Optional[str] = None,
                 config: Optional[RAGPipelineConfig] = None):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: Vector store instance
            llm_repo: Legacy LLM repo specification
            config: Modern RAGPipelineConfig for modular setup
        """
        if config is None:
            config = get_default_config()
        
        self.config = config
        
        # Setup embedding service
        self.embedding_service = create_embedding_service(
            use_distributed=config.use_distributed,
            embedding_config=config.embedding,
            rabbitmq_config=config.rabbitmq
        )
        
        # Setup vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            from src.abstract_vectorstore import create_vector_store
            self.vector_store = create_vector_store(config.vector_store)
        
        # Setup LLM
        if llm_repo:
            # Legacy LLM setup
            token = os.getenv('HF_API_TOKEN')
            endpoint = HuggingFaceEndpoint(repo_id=llm_repo, huggingfacehub_api_token=token)
            self.llm = ChatHuggingFace(llm=endpoint)
        else:
            # Modular LLM setup
            self.llm = create_llm(config.llm)
        
        # For backward compatibility with old embedding manager
        self.embdding_manager = EmbeddingPipeline()

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Search and generate response using RAG
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            Generated response
        """
        results = self.retrieve(query, top_k)
        context = "\n\n".join([doc['content'] for doc in results]) if results else ""
        if not context:
            return "No relevant context found for this query"
        
        if isinstance(self.llm, BaseLLM):
            # Modular LLM interface
            return self.llm.invoke_with_context(context, query)
        else:
            # Legacy LLM interface
            prompt = f"""Use the following context to answer the query
        Context:
        {context}
        
        Question:{query}

        Answer:"""
            response = self.llm.invoke(prompt)
            return response.content
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embeddings([query])[0]
        
        try:
            # Query vector store
            if isinstance(self.vector_store, BaseVectorStore):
                # Modular vector store interface
                results = self.vector_store.query(query_embedding, top_k)
                retrieved_docs = results.get('results', [])
            else:
                # Legacy CHROMAVectorStore interface
                results = self.vector_store.collection.query(
                    query_embeddings=[query_embedding],
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
            
            print(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
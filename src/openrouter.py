"""
OpenRouter API integration for embeddings and LLMs.
Provides access to free and paid models via a unified API.
"""
import requests
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class OpenRouterEmbedding:
    """OpenRouter embedding service using BGE-M3 or other models."""
    
    def __init__(self, api_key: str, model_name: str = "openrouter/bge-m3"):
        """Initialize OpenRouter embedding service.
        
        Args:
            api_key: OpenRouter API key
            model_name: Model identifier (e.g., "openrouter/bge-m3")
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        
        logger.info(f"Initialized OpenRouter embedding with model: {model_name}")
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # Process texts in batches
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self._embed_batch(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                raise
        
        return np.array(embeddings)
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts via OpenRouter API.
        
        Args:
            texts: Batch of texts to embed
            
        Returns:
            List of embedding vectors
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/shamoilaslam/RAG-Basic",
            "X-Title": "RAG Pipeline"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract embeddings from response
            embeddings = []
            if "data" in data:
                # Sort by index to maintain order
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                embeddings = [item["embedding"] for item in sorted_data]
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise


class OpenRouterLLM:
    """OpenRouter LLM service for text generation."""
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "mistral-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """Initialize OpenRouter LLM service.
        
        Args:
            api_key: OpenRouter API key
            model_name: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        
        logger.info(f"Initialized OpenRouter LLM with model: {model_name}")
    
    def invoke(self, prompt: str) -> str:
        """
        Invoke LLM with a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            LLM response
        """
        return self._call_api(prompt)
    
    def invoke_with_context(self, context: str, query: str) -> str:
        """
        Invoke LLM with context and query (RAG style).
        
        Args:
            context: Retrieved context
            query: User query
            
        Returns:
            LLM response
        """
        rag_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question:
{query}

Answer:"""
        return self._call_api(rag_prompt)
    
    def _call_api(self, prompt: str) -> str:
        """
        Call OpenRouter API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            API response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/shamoilaslam/RAG-Basic",
            "X-Title": "RAG Pipeline",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                logger.info(f"Got response from OpenRouter LLM")
                return content
            else:
                logger.error(f"Unexpected API response: {data}")
                raise ValueError("Unexpected response format from OpenRouter API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise


# Free models available on OpenRouter
FREE_MODELS = {
    "embedding": [
        "openrouter/bge-m3",
        "openrouter/bge-large-en-v1-5",
    ],
    "llm": [
        "mistral-7b-instruct",
        "llama-2-7b-chat",
        "neural-chat-7b",
        "nsfwjs/deepdive",
        "google/flan-t5-xl",
    ]
}


def get_available_models() -> dict:
    """Get list of available free models."""
    return FREE_MODELS

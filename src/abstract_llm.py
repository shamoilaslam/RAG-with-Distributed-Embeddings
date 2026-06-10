"""
Abstract LLM interface enabling easy backend swapping.
Supports OpenAI, Anthropic, HuggingFace, and custom implementations.
"""
from abc import ABC, abstractmethod
from typing import Optional
import os
from dotenv import load_dotenv
from src.config import LLMConfig

load_dotenv()


class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Send prompt to LLM and get response
        
        Args:
            prompt: Input prompt
            
        Returns:
            LLM response
        """
        pass
    
    @abstractmethod
    def invoke_with_context(self, context: str, query: str) -> str:
        """
        Invoke LLM with context and query (RAG style)
        
        Args:
            context: Retrieved context
            query: User query
            
        Returns:
            LLM response
        """
        pass


class HuggingFaceLLM(BaseLLM):
    """HuggingFace LLM implementation"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.endpoint = None
        self.llm = None
        self._initialize()
        
    def _initialize(self):
        """Initialize HuggingFace endpoint"""
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        
        token = self.config.api_key or os.getenv("HF_API_TOKEN")
        if not token:
            raise ValueError("HuggingFace API token required")
            
        self.endpoint = HuggingFaceEndpoint(
            repo_id=self.config.model_name,
            huggingfacehub_api_token=token,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens
        )
        self.llm = ChatHuggingFace(llm=self.endpoint)
        print(f"Initialized HuggingFace LLM: {self.config.model_name}")
        
    def invoke(self, prompt: str) -> str:
        """Invoke HuggingFace model"""
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Error invoking HuggingFace LLM: {e}")
            raise
            
    def invoke_with_context(self, context: str, query: str) -> str:
        """Invoke with RAG context"""
        prompt = f"""Use the following context to answer the query.

Context:
{context}

Question: {query}

Answer:"""
        return self.invoke(prompt)


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._initialize()
        
    def _initialize(self):
        """Initialize OpenAI client"""
        from langchain_openai import ChatOpenAI
        
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
            
        self.client = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=api_key
        )
        print(f"Initialized OpenAI LLM: {self.config.model_name}")
        
    def invoke(self, prompt: str) -> str:
        """Invoke OpenAI model"""
        try:
            response = self.client.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Error invoking OpenAI: {e}")
            raise
            
    def invoke_with_context(self, context: str, query: str) -> str:
        """Invoke with RAG context"""
        prompt = f"""Use the following context to answer the query.

Context:
{context}

Question: {query}

Answer:"""
        return self.invoke(prompt)


class AnthropicLLM(BaseLLM):
    """Anthropic LLM implementation"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._initialize()
        
    def _initialize(self):
        """Initialize Anthropic client"""
        from langchain_anthropic import ChatAnthropic
        
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required")
            
        self.client = ChatAnthropic(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=api_key
        )
        print(f"Initialized Anthropic LLM: {self.config.model_name}")
        
    def invoke(self, prompt: str) -> str:
        """Invoke Anthropic model"""
        try:
            response = self.client.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Error invoking Anthropic: {e}")
            raise
            
    def invoke_with_context(self, context: str, query: str) -> str:
        """Invoke with RAG context"""
        prompt = f"""Use the following context to answer the query.

Context:
{context}

Question: {query}

Answer:"""
        return self.invoke(prompt)


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM implementation (free models)"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        from src.openrouter import OpenRouterLLM as ORLLMClient
        
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key required")
            
        self.client = ORLLMClient(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        print(f"Initialized OpenRouter LLM: {config.model_name}")
        
    def invoke(self, prompt: str) -> str:
        """Invoke OpenRouter model"""
        try:
            response = self.client.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error invoking OpenRouter LLM: {e}")
            raise
            
    def invoke_with_context(self, context: str, query: str) -> str:
        """Invoke with RAG context"""
        return self.client.invoke_with_context(context, query)


def create_llm(config: LLMConfig) -> BaseLLM:
    """
    Factory function to create LLM of specified backend
    
    Args:
        config: LLM configuration
        
    Returns:
        Initialized LLM instance
    """
    backend = config.backend.value
    
    if backend == "huggingface":
        return HuggingFaceLLM(config)
    elif backend == "openai":
        return OpenAILLM(config)
    elif backend == "anthropic":
        return AnthropicLLM(config)
    elif backend == "openrouter":
        return OpenRouterLLM(config)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")

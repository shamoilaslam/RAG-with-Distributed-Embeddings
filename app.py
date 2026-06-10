"""
FastAPI application for distributed RAG pipeline.
Supports async query processing with modular backends.
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import tempfile
import os
from pathlib import Path
from src.search import RAGRetriever
from src.config import RAGPipelineConfig, get_default_config, EmbeddingBackend, LLMBackend, VectorStoreBackend
from src.abstract_vectorstore import create_vector_store
from src.data_loader import load_docs
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Distributed RAG Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
rag_retriever: Optional[RAGRetriever] = None
config: RAGPipelineConfig = get_default_config()


class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str = Field(..., description="The user query")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=50)


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    query: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    model_used: str


class ConfigUpdateRequest(BaseModel):
    """Request to update pipeline configuration"""
    embedding_backend: Optional[EmbeddingBackend] = None
    embedding_model: Optional[str] = None
    llm_backend: Optional[LLMBackend] = None
    llm_model: Optional[str] = None
    vector_store_backend: Optional[VectorStoreBackend] = None
    use_distributed: Optional[bool] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vector_store_documents: int
    use_distributed: bool


class IngestResponse(BaseModel):
    """Response model for document ingestion"""
    message: str
    document_count: int
    chunk_count: int
    filenames: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_retriever, config
    logger.info("Initializing RAG pipeline...")
    
    try:
        # Create vector store
        vector_store = create_vector_store(config.vector_store)
        vector_store.load()
        
        # Create RAG retriever with modular components
        rag_retriever = RAGRetriever(config=config)
        rag_retriever.vector_store = vector_store
        
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rag_retriever
    logger.info("Shutting down RAG pipeline...")
    if rag_retriever and hasattr(rag_retriever.embedding_service, 'close'):
        rag_retriever.embedding_service.close()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check pipeline health"""
    if rag_retriever is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    doc_count = rag_retriever.vector_store.get_document_count() if hasattr(rag_retriever.vector_store, 'get_document_count') else 0
    
    return HealthResponse(
        status="healthy",
        vector_store_documents=doc_count,
        use_distributed=config.use_distributed
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Execute RAG query"""
    if rag_retriever is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Retrieve relevant documents
        retrieved_docs = rag_retriever.retrieve(request.query, request.top_k)
        
        # Generate response
        response = rag_retriever.search(request.query, request.top_k)
        
        # Determine model used
        model_used = f"{config.llm.backend.value}:{config.llm.model_name}"
        
        return QueryResponse(
            query=request.query,
            response=response,
            retrieved_documents=retrieved_docs,
            model_used=model_used
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Upload and ingest PDF documents"""
    if rag_retriever is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    temp_dir = tempfile.mkdtemp()
    saved_files = []
    
    try:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            file_path = Path(temp_dir) / file.filename
            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)
            saved_files.append(file.filename)
        
        from src.data_loader import load_docs
        docs = load_docs(temp_dir)
        
        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from PDFs")
        
        chunks = rag_retriever.embedding_service.chunk_documents(docs)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = rag_retriever.embedding_service.generate_embeddings(texts)
        rag_retriever.vector_store.add_documents(chunks, embeddings)
        
        return IngestResponse(
            message="Documents ingested successfully",
            document_count=len(docs),
            chunk_count=len(chunks),
            filenames=saved_files
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/config")
async def get_config():
    """Get current pipeline configuration"""
    return {
        "embedding": {
            "backend": config.embedding.backend.value,
            "model": config.embedding.model_name,
            "batch_size": config.embedding.batch_size
        },
        "vector_store": {
            "backend": config.vector_store.backend.value,
            "collection": config.vector_store.collection_name
        },
        "llm": {
            "backend": config.llm.backend.value,
            "model": config.llm.model_name,
            "temperature": config.llm.temperature
        },
        "distributed": {
            "enabled": config.use_distributed,
            "rabbitmq_host": config.rabbitmq.host,
            "workers": config.rabbitmq.num_workers
        }
    }


@app.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """Update pipeline configuration (note: requires restart for some changes)"""
    global config
    
    # Update embedding config
    if request.embedding_backend:
        config.embedding.backend = request.embedding_backend
    if request.embedding_model:
        config.embedding.model_name = request.embedding_model
    
    # Update LLM config
    if request.llm_backend:
        config.llm.backend = request.llm_backend
    if request.llm_model:
        config.llm.model_name = request.llm_model
    
    # Update vector store config
    if request.vector_store_backend:
        config.vector_store.backend = request.vector_store_backend
    
    # Update distributed flag
    if request.use_distributed is not None:
        config.use_distributed = request.use_distributed
    
    logger.info(f"Configuration updated: {request}")
    return {"message": "Configuration updated. Restart app for embedding/LLM changes to take effect."}


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    if rag_retriever is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stats = {
        "total_documents": rag_retriever.vector_store.get_document_count() if hasattr(rag_retriever.vector_store, 'get_document_count') else 0,
        "use_distributed": config.use_distributed,
        "embedding_model": config.embedding.model_name,
        "llm_model": config.llm.model_name,
        "vector_store_backend": config.vector_store.backend.value
    }
    
    # Add RabbitMQ stats if distributed
    if config.use_distributed and hasattr(rag_retriever.embedding_service, 'pipeline'):
        rabbitmq_stats = rag_retriever.embedding_service.pipeline.get_queue_stats()
        stats["rabbitmq"] = rabbitmq_stats
    
    return stats


@app.get("/documents")
async def list_documents():
    """List document metadata from vector store"""
    if rag_retriever is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        count = rag_retriever.vector_store.get_document_count()
        return {"documents": [{"id": f"collection-1", "count": count, "name": config.vector_store.collection_name}]}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from vector store"""
    if rag_retriever is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        rag_retriever.vector_store.delete_collection()
        rag_retriever.vector_store = create_vector_store(config.vector_store)
        rag_retriever.vector_store.load()
        return {"message": "All documents cleared", "document_count": 0}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
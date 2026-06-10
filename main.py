"""
Main entry point demonstrating local usage of distributed RAG pipeline.
Shows modular component swapping and distributed vs local modes.
"""
from src.config import get_default_config, RAGPipelineConfig, EmbeddingBackend, LLMBackend, VectorStoreBackend
from src.search import RAGRetriever
from src.abstract_vectorstore import create_vector_store
from src.data_loader import load_docs
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_local_pipeline(use_distributed: bool = False):
    """
    Setup RAG pipeline with modular components
    
    Args:
        use_distributed: Whether to use distributed embedding via RabbitMQ
        
    Returns:
        RAGRetriever instance
    """
    logger.info(f"Setting up RAG pipeline (distributed={use_distributed})")
    
    # Get configuration
    config = get_default_config()
    config.use_distributed = use_distributed
    
    # Create vector store
    vector_store = create_vector_store(config.vector_store)
    vector_store.load()
    logger.info(f"Vector store initialized: {config.vector_store.backend.value}")
    
    # Create RAG retriever with modular components
    retriever = RAGRetriever(config=config)
    retriever.vector_store = vector_store
    
    logger.info(f"RAG pipeline ready")
    logger.info(f"  Embedding: {config.embedding.backend.value}:{config.embedding.model_name}")
    logger.info(f"  Vector Store: {config.vector_store.backend.value}")
    logger.info(f"  LLM: {config.llm.backend.value}:{config.llm.model_name}")
    
    return retriever


def ingest_documents(retriever: RAGRetriever, data_directory: str = './data'):
    """
    Ingest documents into vector store
    
    Args:
        retriever: RAGRetriever instance
        data_directory: Path to PDF documents
    """
    logger.info(f"Loading documents from {data_directory}...")
    
    try:
        docs = load_docs(data_directory)
        if not docs:
            logger.warning("No documents found")
            return
        
        logger.info(f"Loaded {len(docs)} document pages")
        
        # Chunk documents
        chunks = retriever.embedding_service.chunk_documents(docs)
        logger.info(f"Chunked into {len(chunks)} segments")
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = retriever.embedding_service.generate_embeddings(texts)
        logger.info(f"Generated embeddings for {len(embeddings)} chunks")
        
        # Add to vector store
        retriever.vector_store.add_documents(chunks, embeddings)
        logger.info(f"Documents ingested successfully")
        
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}", exc_info=True)


def interactive_search(retriever: RAGRetriever):
    """
    Interactive search loop
    
    Args:
        retriever: RAGRetriever instance
    """
    logger.info("Starting interactive search (type 'quit' to exit)...")
    
    while True:
        try:
            query = input("\n📝 Enter query: ").strip()
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            logger.info(f"Searching for: {query}")
            response = retriever.search(query, top_k=5)
            
            print(f"\n🤖 Response:\n{response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during search: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Distributed RAG Pipeline")
    parser.add_argument('--mode', choices=['ingest', 'search', 'both'], default='search',
                       help='Operation mode')
    parser.add_argument('--data-dir', default='./data',
                       help='Path to data directory for ingestion')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed embedding via RabbitMQ')
    parser.add_argument('--query', type=str,
                       help='Single query (non-interactive)')
    
    args = parser.parse_args()
    
    try:
        # Setup pipeline
        retriever = setup_local_pipeline(use_distributed=args.distributed)
        
        # Execute operations
        if args.mode in ['ingest', 'both']:
            ingest_documents(retriever, args.data_dir)
        
        if args.mode in ['search', 'both']:
            if args.query:
                # Single query mode
                logger.info(f"Executing query: {args.query}")
                response = retriever.search(args.query, top_k=5)
                print(f"\n🤖 Response:\n{response}")
            else:
                # Interactive mode
                interactive_search(retriever)
        
        logger.info("Pipeline execution completed")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

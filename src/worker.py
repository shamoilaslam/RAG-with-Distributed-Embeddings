"""
RabbitMQ worker service for processing embedding tasks.
Run multiple instances to enable parallel processing and latency reduction.
"""
import json
import logging
import signal
import sys
from typing import Optional
import pika
import numpy as np
from src.config import EmbeddingConfig, RabbitMQConfig
from src.embedding import EmbeddingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """
    Consumer that processes embedding tasks from RabbitMQ queue.
    Handles automatic reconnection and graceful shutdown.
    """
    
    def __init__(
        self, 
        embedding_config: EmbeddingConfig,
        rabbitmq_config: RabbitMQConfig,
        worker_id: Optional[str] = None
    ):
        self.embedding_config = embedding_config
        self.rabbitmq_config = rabbitmq_config
        self.worker_id = worker_id or f"worker_{np.random.randint(10000)}"
        self.connection = None
        self.channel = None
        self.embedding_pipeline = None
        self.should_stop = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle graceful shutdown"""
        logger.info(f"[{self.worker_id}] Received shutdown signal")
        self.should_stop = True
        if self.channel:
            self.channel.stop_consuming()
            
    def connect(self):
        """Establish connection to RabbitMQ"""
        credentials = pika.PlainCredentials(
            self.rabbitmq_config.username, 
            self.rabbitmq_config.password
        )
        parameters = pika.ConnectionParameters(
            host=self.rabbitmq_config.host,
            port=self.rabbitmq_config.port,
            virtual_host=self.rabbitmq_config.vhost,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        # Set prefetch to limit concurrent messages
        self.channel.basic_qos(prefetch_count=self.rabbitmq_config.prefetch_count)
        
        # Declare queue
        self.channel.queue_declare(
            queue=self.rabbitmq_config.embedding_queue,
            durable=True
        )
        
        logger.info(f"[{self.worker_id}] Connected to RabbitMQ")
        
    def initialize_embedding_model(self):
        """Lazy load embedding model on first use"""
        if self.embedding_pipeline is None:
            logger.info(f"[{self.worker_id}] Initializing embedding model: {self.embedding_config.model_name}")
            self.embedding_pipeline = EmbeddingPipeline(
                model_name=self.embedding_config.model_name,
                chunk_size=self.embedding_config.chunk_size,
                chunk_overlap=self.embedding_config.chunk_overlap
            )
            
    def process_message(self, ch, method, properties, body):
        """
        Callback for processing embedding tasks.
        
        Args:
            ch: Channel
            method: Method frame
            properties: Properties frame
            body: Message body
        """
        task_id = properties.correlation_id
        logger.info(f"[{self.worker_id}] Processing task {task_id}")
        
        try:
            # Parse task
            task_data = json.loads(body.decode())
            texts = task_data.get('texts', [])
            callback_queue = task_data.get('callback_queue')
            
            if not texts:
                logger.warning(f"[{self.worker_id}] Received empty text list in task {task_id}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
                
            # Initialize model if needed
            self.initialize_embedding_model()
            
            # Generate embeddings
            logger.info(f"[{self.worker_id}] Embedding {len(texts)} texts")
            embeddings = self.embedding_pipeline.generate_embeddings(texts)
            
            # Prepare result
            result = {
                "task_id": task_id,
                "status": "completed",
                "embeddings": embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                "text_count": len(texts),
                "worker_id": self.worker_id
            }
            
            # Send result back if callback queue specified
            if callback_queue:
                self._send_result(result, callback_queue, task_id)
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"[{self.worker_id}] Completed task {task_id}")
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error processing task {task_id}: {str(e)}", exc_info=True)
            # Reject and requeue
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
    def _send_result(self, result: dict, callback_queue: str, task_id: str):
        """Send result to callback queue"""
        try:
            message = json.dumps(result)
            self.channel.basic_publish(
                exchange='',
                routing_key=callback_queue,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,
                    correlation_id=task_id
                )
            )
            logger.info(f"[{self.worker_id}] Sent result to callback queue {callback_queue}")
        except Exception as e:
            logger.error(f"[{self.worker_id}] Failed to send result: {str(e)}")
            
    def start(self):
        """Start consuming and processing tasks"""
        try:
            self.connect()
            
            # Register callback
            self.channel.basic_consume(
                queue=self.rabbitmq_config.embedding_queue,
                on_message_callback=self.process_message
            )
            
            logger.info(f"[{self.worker_id}] Starting to consume from {self.rabbitmq_config.embedding_queue}")
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info(f"[{self.worker_id}] Interrupted")
        except Exception as e:
            logger.error(f"[{self.worker_id}] Fatal error: {str(e)}", exc_info=True)
        finally:
            self.close()
            
    def close(self):
        """Close connection"""
        if self.channel:
            self.channel.stop_consuming()
        if self.connection:
            self.connection.close()
        logger.info(f"[{self.worker_id}] Closed connection")


def main():
    """Entry point for worker process"""
    from src.config import get_default_config
    
    config = get_default_config()
    worker = EmbeddingWorker(
        embedding_config=config.embedding,
        rabbitmq_config=config.rabbitmq
    )
    worker.start()


if __name__ == "__main__":
    main()

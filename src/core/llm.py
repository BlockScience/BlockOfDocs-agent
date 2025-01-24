from typing import Optional
import requests
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils.logger import setup_logger
from utils.config import load_config

logger = setup_logger()
config = load_config()

class OllamaLLM:
    def __init__(self, model_name: str = None):
        self.llm = Ollama(
            model=model_name or config.MODEL_NAME,
            base_url=config.OLLAMA_API_URL,
            request_timeout=120.0
        )
        self.embedding_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Generate a response using the Ollama API."""
        try:
            response = self.llm.complete(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"Successfully generated response for prompt: {prompt[:50]}...")
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def get_embeddings(self, text: str) -> list:
        """Get embeddings for text using HuggingFace."""
        try:
            embeddings = self.embedding_model.get_text_embedding(
                text
            )
            logger.info(f"Successfully generated embeddings for text: {text[:50]}...")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

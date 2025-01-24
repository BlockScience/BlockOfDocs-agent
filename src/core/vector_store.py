from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils.logger import setup_logger
from utils.config import load_config
from core.llm import OllamaLLM

logger = setup_logger()
config = load_config()

class VectorStore:
    def __init__(self, docs_dir: str = "./data/raw"):
        """Initialize the vector store with improved LlamaIndex configuration."""
        self._docs_dir = Path(docs_dir)
        self._query_engine = None
        
        # Ensure docs directory exists
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize if documents exist
        if list(self._docs_dir.glob("*.md")):
            self.setup_llamaindex()
            
    def setup_llamaindex(self):
        """Setup LlamaIndex with optimized models and custom prompt template."""
        try:
            # Setup LLM with increased timeout
            ollama = OllamaLLM()
            
            # Setup advanced embedding model
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-en-v1.5",
                trust_remote_code=True
            )
            
            # Configure global settings
            Settings.embed_model = embed_model
            Settings.llm = ollama.llm  # Use the llm property from OllamaLLM instance
            
            # Load documents with specific extensions
            loader = SimpleDirectoryReader(
                input_dir=self._docs_dir,
                required_exts=[".md"],
                recursive=True
            )
            docs = loader.load_data()
            
            if not docs:
                logger.warning("No markdown documents found in the specified directory")
                return
                
            # Create optimized index
            logger.info("Creating vector store index...")
            index = VectorStoreIndex.from_documents(
                docs,
                show_progress=True
            )
            
            # Setup streaming query engine with custom prompt
            qa_prompt_tmpl = PromptTemplate(
                (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step "
                    "to answer the query in a crisp manner, in case you don't know the "
                    "answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                )
            )
            
            # Configure query engine with streaming and custom prompt
            self._query_engine = index.as_query_engine(
                streaming=True,
                similarity_top_k=3
            )
            self._query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
            )
            
            logger.info("Successfully initialized vector store with custom configuration")
            
        except Exception as e:
            logger.error(f"Error setting up LlamaIndex: {str(e)}")
            raise
            
    def query(self, query_text: str) -> str:
        """
        Query the vector store with streaming response.
        
        Args:
            query_text: The query string
            
        Returns:
            Generator for streaming response
        """
        if not self._query_engine:
            raise ValueError("Vector store not initialized. Please ensure documents are loaded.")
            
        try:
            response = self._query_engine.query(query_text)
            return response
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise
            
    def clear(self):
        """Reset the vector store and query engine."""
        self._query_engine = None
        logger.info("Vector store cleared")

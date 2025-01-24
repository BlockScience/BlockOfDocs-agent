from typing import List
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from utils.logger import setup_logger

logger = setup_logger()

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def process_file(self, file_path: Path) -> List[Document]:
        """Process a single file and return chunked documents."""
        try:
            reader = SimpleDirectoryReader(input_files=[str(file_path)])
            documents = reader.load_data()
            logger.info(f"Successfully loaded document: {file_path}")
            
            # Parse documents into nodes/chunks
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} chunks from document")
            
            return nodes
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def process_directory(self, directory_path: Path) -> List[Document]:
        """Process all markdown files in a directory."""
        try:
            reader = SimpleDirectoryReader(
                input_dir=str(directory_path),
                recursive=True,
                required_exts=[".md"]
            )
            documents = reader.load_data()
            logger.info(f"Successfully loaded documents from {directory_path}")
            
            # Parse documents into nodes/chunks
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} total chunks from all documents")
            
            return nodes
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise

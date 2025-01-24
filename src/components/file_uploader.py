import streamlit as st
from pathlib import Path
import tempfile
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from utils.logger import setup_logger
from utils.config import load_config

logger = setup_logger()
config = load_config()

def render_file_uploader():
    """Render the file upload interface."""
    st.subheader("📄 Upload Documents")
    
    # Initialize document processor and vector store
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload markdown files",
        type=["md"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            try:
                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    # Check file size
                    if uploaded_file.size > config.MAX_FILE_SIZE:
                        st.error(f"File {uploaded_file.name} is too large. Maximum size is {config.MAX_FILE_SIZE/1024/1024:.1f}MB")
                        continue
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)
                    
                    # Process document
                    nodes = doc_processor.process_file(tmp_path)
                    
                    # Add to vector store
                    vector_store.add_documents(nodes)
                    
                    # Clean up temporary file
                    tmp_path.unlink()
                
                st.success(f"Successfully processed {len(uploaded_files)} documents!")
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                logger.error(f"Error in file upload: {str(e)}")
    
    # Display currently indexed documents
    with st.expander("📚 Indexed Documents"):
        st.info("List of currently indexed documents will appear here")
        # Add functionality to list indexed documents

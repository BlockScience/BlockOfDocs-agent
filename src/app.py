import streamlit as st
from components.sidebar import render_sidebar
from components.file_uploader import render_file_uploader
from components.query_interface import render_query_interface
from utils.config import load_config
from utils.logger import setup_logger

# Setup logging
logger = setup_logger()

# Load configuration
config = load_config()

def main():
    st.set_page_config(
        page_title="RAG Document Search",
        page_icon="🔍",
        layout="wide"
    )
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("Document Search with RAG")
    
    # File upload section
    render_file_uploader()
    
    # Query interface
    render_query_interface()

if __name__ == "__main__":
    main()

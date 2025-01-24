import streamlit as st
from utils.config import load_config

config = load_config()

def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        st.selectbox(
            "Model",
            ["mistral", "llama2", "codellama"],
            index=0,
            key="model_name"
        )
        
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature"
        )
        
        st.slider(
            "Max Tokens",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
            key="max_tokens"
        )
        
        # Search settings
        st.subheader("Search Configuration")
        st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="top_k"
        )
        
        # System info
        st.subheader("System Info")
        st.text(f"API URL: {config.OLLAMA_API_URL}")
        st.text(f"Vector Store: {config.CHROMA_PERSIST_DIR}")
        
        # Clear data button
        if st.button("Clear All Data", type="secondary"):
            # Add clear data functionality
            st.warning("This will delete all indexed documents.")
            # Add confirmation dialog

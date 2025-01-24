import streamlit as st
from core.vector_store import VectorStore
from core.llm import OllamaLLM
from utils.logger import setup_logger

logger = setup_logger()

def render_query_interface():
    """Render the query interface for searching documents."""
    st.subheader("🔍 Search Documents")
    
    # Initialize components
    vector_store = VectorStore()
    llm = OllamaLLM()
    
    # Query input
    query = st.text_area(
        "Enter your question",
        placeholder="What would you like to know about the documents?",
        key="query_input"
    )
    
    if st.button("Search", type="primary"):
        if not query:
            st.warning("Please enter a question.")
            return
        
        try:
            with st.spinner("Searching..."):
                # Get relevant documents
                response = vector_store.query(query)
                
                # Generate response using LLM
                context = "\n".join([doc.text for doc in response.source_nodes])
                llm_response = llm.generate(
                    prompt=query,
                    context=context,
                    temperature=st.session_state.get("temperature", 0.7),
                    max_tokens=st.session_state.get("max_tokens", 512)
                )
                
                # Display results
                st.markdown("### Answer")
                st.write(llm_response)
                
                # Display sources
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(response.source_nodes, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(doc.text)
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Error in query interface: {str(e)}")
    
    # Display query history
    with st.expander("📝 Query History"):
        st.info("Recent queries will appear here")
        # Add functionality to show query history

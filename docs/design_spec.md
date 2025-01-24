## Overview

This project is an intelligent query interface for the "Blocks of Docs" framework, built using Streamlit, LlamaIndex, and Ollama LLM. The system is designed to help users navigate and interact with Complex Blocks (stored as markdown files) from the Blocks of Docs framework. Using LlamaIndex's document processing pipeline, it ingests these Block definitions and stores their embeddings in a local Chroma DB, enabling semantic search and intelligent retrieval. When users need to find appropriate Blocks for their documentation needs, they can query the system through a Streamlit frontend, and the VectorStoreQueryEngine (powered by Mistral-7B via Ollama) processes these queries to recommend the most relevant Blocks from the framework. The system acts as an intelligent librarian for Blocks of Docs, helping users navigate the complex ecosystem of document templates and requirements by providing context-aware search, recommendations, and insights into the various Block types available in the framework. This creates an efficient interface for discovering and understanding the appropriate Blocks for different documentation needs, while maintaining the integrity and purpose of the original Blocks of Docs framework.

## Technology Stack Overview

### 1. Streamlit (Frontend)

- **Implementation Details**:
  - Using `st.file_uploader` for markdown file ingestion
  - `st.text_input` and `st.text_area` for query interface
  - `st.spinner` and `st.progress` for loading states
  - Session state management for handling document and query history
  - Markdown rendering using `st.markdown` for displaying results
  - Caching with `@st.cache_data` for performance optimization

### 2. LlamaIndex (Document Processing & RAG)

- **Core Components**:

  - **Document Loading**:

    - Using `SimpleDirectoryReader` for markdown file processing
    - Custom metadata extraction for file attributes

  - **Text Chunking**:

    - `SentenceSplitter` with chunk size of 1024 tokens
    - Overlap of 200 tokens for context preservation

  - **Vector Store**:

    - Local Chroma DB instance for document embeddings
    - Dimension: 384 (using all-MiniLM-L6-v2 embeddings)
    - Metadata filtering for efficient retrieval

  - **Query Engine**:
    - Using `VectorStoreQueryEngine` with similarity search
    - Top-k retrieval (k=3) for context gathering
    - Re-ranking based on relevance scores

### 3. Ollama (Local LLM Integration)

- **Setup**:
  - Running Mistral-7B model locally
  - API endpoint: http://localhost:11434
  - Batch processing for efficient inference
- **API Integration**:
  - Using REST API for model interactions
  - Endpoints:
    - `/api/generate` for completion generation
    - `/api/embeddings` for text embeddings
- **Parameters**:
  - Temperature: 0.7 for balanced creativity
  - Max tokens: 512 for responses
  - Context window: 8192 tokens

## Data Flow

1. **Document Ingestion**:

   ```
   Markdown Files → Streamlit Upload → LlamaIndex Processing → Chroma DB
   ```

2. **Query Processing**:

   ```
   User Query → Vector Search → Context Retrieval → Ollama LLM → Response Generation
   ```

3. **Response Flow**:
   ```
   LLM Output → Post-processing → Markdown Rendering → UI Display
   ```

## Simplified User Interaction Workflow

1. **Identify Needs:**

   - **User Input:** Users describe their document needs through an interface (e.g., Slack or web portal).
   - **Initial Assessment:** The Librarian bot analyzes the input to understand the required document's scope.

2. **Select Blocks:**

   - **Recommendations:** The Librarian suggests suitable Blocks, prioritizing simple ones for straightforward needs and complex options for intricate requirements.
   - **Alternatives:** If multiple options apply, the Librarian provides brief descriptions to help users choose.

3. **Review Requirements:**

   - **Detailed Overview:** Users can request a full description of selected Blocks, including their structure and criteria.
   - **Customization:** Blocks can be adjusted to better match specific needs.

4. **Answer Requirements:**

   - **Interactive Prompts:** The Librarian guides users through answering questions based on the selected Blocks.
   - **Organized Responses:** User answers are compiled coherently, ensuring completeness.
   - **Validation:** The Librarian checks responses for consistency and helps refine inputs as needed.

5. **Compile and Format Document:**

   - **Content Compilation:** The Librarian assembles information into a cohesive document.
   - **Template Formatting:** Content is formatted according to predefined styles and criteria for professionalism.
   - **Preview:** Users can review and request adjustments before finalization.

6. **Finalize and Deliver Document:**
   - **Delivery:** The finalized document is sent directly to users via Slack, ensuring prompt and seamless access.

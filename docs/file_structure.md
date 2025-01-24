# Project File Structure

```
📦 rag-app/
├── 📁 src/
│   ├── 📁 components/
│   │   ├── 📄 sidebar.py         # Streamlit sidebar component
│   │   ├── 📄 file_uploader.py   # File upload component
│   │   └── 📄 query_interface.py # Search interface component
│   │
│   ├── 📁 core/
│   │   ├── 📄 document_processor.py  # LlamaIndex document processing
│   │   ├── 📄 embeddings.py         # Vector embeddings logic
│   │   ├── 📄 llm.py               # Ollama LLM integration
│   │   └── 📄 vector_store.py      # ChromaDB operations
│   │
│   ├── 📁 utils/
│   │   ├── 📄 config.py           # Configuration management
│   │   ├── 📄 logger.py           # Logging setup
│   │   └── 📄 helpers.py          # Utility functions
│   │
│   └── 📄 app.py                  # Main Streamlit application
│
├── 📁 tests/
│   ├── 📄 test_document_processor.py
│   ├── 📄 test_embeddings.py
│   └── 📄 test_llm.py
│
├── 📁 data/
│   ├── 📁 raw/                    # Original markdown files
│   ├── 📁 processed/              # Processed documents
│   └── 📁 vector_store/           # ChromaDB storage
│
├── 📁 docs/
│   ├── 📄 spec.md                 # System specification
│   ├── 📄 design_spec.md          # Technical design
│   └── 📄 api.md                  # API documentation
│
├── 📄 requirements.txt            # Python dependencies
├── 📄 .env.example               # Environment variables template
├── 📄 .gitignore                 # Git ignore rules
├── 📄 README.md                  # Project documentation
└── 📄 Makefile                   # Build and run commands
```

## Key Components Description

### Source Code (`src/`)

1. **components/**

   - Modular Streamlit UI components
   - Each component is a self-contained feature
   - Reusable across different pages

2. **core/**

   - Core business logic and data processing
   - LlamaIndex integration
   - Ollama LLM setup
   - Vector store operations

3. **utils/**
   - Shared utilities and helpers
   - Configuration management
   - Logging setup

### Tests (`tests/`)

- Unit tests for core functionality
- Integration tests for LLM and vector store
- Test fixtures and utilities

### Data (`data/`)

- Organized data pipeline flow
- Separate raw and processed data
- Vector store persistence

### Documentation (`docs/`)

- Comprehensive project documentation
- API specifications
- Design documents

## Configuration Files

1. **requirements.txt**

```
streamlit==1.31.0
llama-index==0.9.7
chromadb==0.4.22
python-dotenv==1.0.0
pytest==7.4.4
```

2. **.env.example**

```
OLLAMA_API_URL=http://localhost:11434
CHROMA_PERSIST_DIR=./data/vector_store
MAX_FILE_SIZE=10485760  # 10MB in bytes
LOG_LEVEL=INFO
```

3. **Makefile**

```makefile
.PHONY: setup test run clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

test:
	pytest tests/

run:
	streamlit run src/app.py

clean:
	rm -rf data/processed/*
	rm -rf data/vector_store/*
```

## Usage

1. Clone the repository
2. Copy `.env.example` to `.env` and configure
3. Run `make setup` to install dependencies
4. Run `make run` to start the application
5. Access the app at `http://localhost:8501`

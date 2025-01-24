# RAG Document Search Application

A Streamlit-based application for semantic document search using LlamaIndex and Ollama.

## Features

- Upload and process markdown documents
- Semantic search using RAG (Retrieval Augmented Generation)
- Local LLM integration via Ollama
- Vector storage with ChromaDB
- Interactive UI with Streamlit

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Set up the environment:
```bash
make setup
```

3. Configure the application:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

1. Start the application:
```bash
make run
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload markdown documents and start searching!

## Development

- Run tests: `make test`
- Format code: `make format`
- Lint code: `make lint`
- Clean data: `make clean`

## Project Structure

```
📦 rag-app/
├── src/              # Source code
├── tests/            # Test files
├── data/             # Data storage
└── docs/             # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

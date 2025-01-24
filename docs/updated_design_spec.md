# Updated Technical Design Specification

## 1. System Architecture

### 1.1 Frontend (Reflex)

- **Components**:
  - `QueryInterface`: Main search interface with real-time query processing
  - `FileUploader`: Document upload component with validation
  - `Sidebar`: Configuration panel for model and system settings
  - `Layout`: Responsive layout system with modern UI/UX

### 1.2 Core Services

- **Document Processing**:

  - Chunk size: 1024 tokens
  - Overlap: 200 tokens
  - SentenceSplitter for semantic chunking
  - Support for markdown and text files
  - File size limit: 10MB

- **Vector Store (LlamaIndex)**:

  - Built-in vector store with optimized indexing
  - Streaming response support
  - Custom prompt templates
  - Automatic document validation
  - Improved error handling

- **LLM Integration**:

  - Primary: Ollama API integration with deepseek-r1:8b
  - Enhanced embedding model: BAAI/bge-large-en-v1.5
  - Configurable parameters:
    - Temperature: 0.0-1.0 (default: 0.7)
    - Max tokens: 512 (configurable)
    - Request timeout: 120s
  - Streaming response support
  - Custom prompt templates

- **Embedding System**:
  - Model: BAAI/bge-large-en-v1.5
  - Enhanced semantic understanding
  - Trust remote code support
  - Batch processing optimization
  - Hugging Face integration

## 2. Data Flow

### 2.1 Document Ingestion

```
Upload → Validation → Chunking → Embedding → Vector Storage
```

### 2.2 Query Processing

```
Query → Embedding → Similarity Search → Context Retrieval → LLM Processing → Response
```

### 2.3 System State Management

```
Configuration → Environment Variables → Runtime Settings → Component State
```

## 3. Technical Components

### 3.1 Core Classes

- **VectorStore**:

  - Manages document storage and retrieval
  - Handles collection persistence
  - Provides CRUD operations
  - Implements similarity search

- **DocumentProcessor**:

  - Handles file processing and chunking
  - Manages document validation
  - Implements metadata extraction
  - Supports batch processing

- **OllamaLLM**:
  - Manages LLM interactions
  - Handles response generation
  - Provides embedding services
  - Implements retry logic

### 3.2 Configuration Management

- **Settings**:
  - Environment-based configuration
  - Type-safe settings with Pydantic
  - Centralized configuration management
  - Dynamic setting updates

### 3.3 Utilities

- **Logger**:
  - Structured logging
  - Level-based filtering
  - Console and file outputs
  - Error tracking

## 4. Security Measures

### 4.1 File Security

- Size limitations (10MB max)
- Type validation
- Content sanitization
- Secure storage practices

### 4.2 API Security

- Rate limiting
- Request validation
- Error handling
- Timeout management

## 5. Performance Optimizations

### 5.1 Caching

- Vector store persistence
- Embedding caching
- Query result caching
- Configuration caching

### 5.2 Processing

- Batch document processing
- Async operations
- Efficient chunking
- Optimized search

## 6. Development Guidelines

### 6.1 Code Organization

- Modular component structure
- Clear separation of concerns
- Type hints and documentation
- Consistent error handling

### 6.2 Testing Strategy

- Unit tests for core components
- Integration tests for workflows
- Performance benchmarks
- Error scenario coverage

## 7. Deployment Considerations

### 7.1 Dependencies

- Python 3.9+
- LlamaIndex
- Ollama
- Reflex
- Required environment variables

### 7.2 Infrastructure

- Local development setup
- Production deployment options
- Scaling considerations
- Monitoring requirements

## 8. Future Enhancements

### 8.1 Planned Features

- Multi-model support
- Advanced query optimization
- Enhanced metadata management
- Improved error recovery

### 8.2 Technical Debt

- Configuration refactoring
- Test coverage expansion
- Documentation updates
- Performance optimization

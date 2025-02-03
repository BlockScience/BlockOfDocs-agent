# GraphRAG

A knowledge graph-based RAG (Retrieval-Augmented Generation) system with Slack integration.

## Features

- Document processing and knowledge extraction
- Neo4j-based knowledge graph storage
- Community detection and summarization
- Slack bot integration for querying
- Async processing support

## Setup

1. Clone the repository
2. Copy .env.example to .env and configure your environment variables
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Start Neo4j database
5. Run the application:
   ```bash
   poetry run python src/main.py
   ```

## Architecture

- Core components for knowledge graph operations
- Slack integration for user interaction
- Modular design for easy extension
- Async processing for better performance

````mermaid
flowchart TB
    subgraph Input Sources
        MD[Markdown Files]
        JSON[JSON Manifests]
        TXT[Text Blocks]
        SLACK[Slack Messages]
    end

    subgraph Document Processing
        DL[DocumentLoader]
        DT[DocumentTracker]
        SP[SentenceSplitter]
    end

    subgraph Knowledge Graph Processing
        KGE[GraphRAGExtractor]
        GS[GraphRAGStore]
        PGI[PropertyGraphIndex]
    end

    subgraph Query System
        QE[GraphRAGQueryEngine]
        LLM[OpenAI LLM]
    end

    subgraph Integration
        SB[SlackBot]
        NEO4J[(Neo4j Database)]
    end

    MD & JSON & TXT --> DL
    DL --> |Track Changes| DT
    DL --> |Split Documents| SP
    SP --> |Extract Entities & Relations| KGE
    KGE --> |Store Graph Data| GS
    GS --> |Build Index| PGI
    GS <--> NEO4J
    PGI --> QE
    QE <--> LLM
    SLACK --> SB
    SB --> |Query| QE
    QE --> |Response| SB
    ```
````

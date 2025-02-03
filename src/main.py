# src/main.py

import asyncio
import nest_asyncio
from llama_index.core import StorageContext, PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from config.settings import (
    NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_URL,
    OPENAI_MODEL, OPENAI_API_KEY,
    SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET
)
from data.loader import DocumentLoader
from core.extractor import GraphRAGExtractor
from core.store import GraphRAGStore
from core.query_engine import GraphRAGQueryEngine
from prompts.templates import KG_TRIPLET_EXTRACT_TMPL, parse_fn
from slack.bot import SlackBot


# Apply nested asyncio support
nest_asyncio.apply()

async def init_knowledge_graph():
    # Initialize document loader with tracking
    doc_loader = DocumentLoader("context")
    documents, changes = doc_loader.load_documents()
    
    # Log document changes
    print("\nDocument Changes:")
    print(f"New documents: {len(changes['new'])}")
    print(f"Modified documents: {len(changes['modified'])}")
    print(f"Unchanged documents: {len(changes['unchanged'])}")
    
    # Initialize storage and graph store
    storage_context = StorageContext.from_defaults()
    graph_store = GraphRAGStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL
    )
    
    if not documents:
        if changes['unchanged']:
            print("\nNo new or modified documents to process.")
            # Create an index with the existing graph store
            index = PropertyGraphIndex.from_documents(
                [],
                storage_context=storage_context,
                property_graph_store=graph_store
            )
            
            llm = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
            query_engine = GraphRAGQueryEngine(
                graph_store=graph_store,
                index=index,
                llm=llm
            )
            
            return index, query_engine
        else:
            raise ValueError("No documents found in the context directory.")
    
    print(f"\nProcessing {len(documents)} documents...")
    
    # Split documents into nodes
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Split into {len(nodes)} nodes for processing...")
    
    # Initialize extractor
    llm = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        parse_fn=parse_fn,
        max_paths_per_chunk=2,
        num_workers=2,
        batch_size=5,
        retry_delay=3.0
    )
    
    try:
        # Process nodes
        nodes = await kg_extractor.acall(nodes, show_progress=True)
        
        # Build new index or update existing
        if changes['unchanged']:
            print("\nMerging new nodes with existing graph...")
            # Add nodes to existing graph and create index
            for node in nodes:
                graph_store.add_node(node)
            index = PropertyGraphIndex.from_documents(
                [],
                storage_context=storage_context,
                property_graph_store=graph_store
            )
        else:
            print("\nBuilding new graph index...")
            index = PropertyGraphIndex(
                nodes=nodes,
                kg_extractors=[kg_extractor],
                property_graph_store=graph_store,
                show_progress=True,
                storage_context=storage_context
            )
        
        # Create query engine
        query_engine = GraphRAGQueryEngine(
            graph_store=graph_store,
            index=index,
            llm=llm
        )
        
        return index, query_engine
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise


def main():
    try:
        # Initialize knowledge graph
        kg_index, query_engine = asyncio.run(init_knowledge_graph())
        print("Knowledge graph initialized successfully.")

        # Initialize and run Slack bot
        if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET:
            raise ValueError("Slack credentials not found in environment variables")

        slack_bot = SlackBot(
            slack_token=SLACK_BOT_TOKEN,
            signing_secret=SLACK_SIGNING_SECRET,
            kg_index=kg_index,
            query_engine=query_engine
        )

        print("Starting Slack bot...")
        slack_bot.run(port=3000)

    except Exception as e:
        print(f"Failed to initialize system: {str(e)}")
        raise

if __name__ == "__main__":
    main()
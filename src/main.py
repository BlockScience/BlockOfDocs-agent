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
from core.extractor import GraphRAGExtractor
from core.store import GraphRAGStore
from core.query_engine import GraphRAGQueryEngine
from data.loader import load_documents
from prompts.templates import KG_TRIPLET_EXTRACT_TMPL, parse_fn
from slack.bot import SlackBot


# Apply nested asyncio support
nest_asyncio.apply()

async def init_knowledge_graph():
    # Load documents
    documents = load_documents("context")  # Make sure this points to your context directory
    if not documents:
        raise ValueError("No documents found. Please check the context directory.")
    
    print(f"Loading {len(documents)} documents...")
    
    # Split documents into nodes
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Split into {len(nodes)} nodes for processing...")
    
    # Initialize extractor with the parse function
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
        
        # Initialize storage and graph store
        storage_context = StorageContext.from_defaults()
        graph_store = GraphRAGStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URL
        )
        
        # Build index
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
            llm=llm,
            index=index,
            similarity_top_k=10
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
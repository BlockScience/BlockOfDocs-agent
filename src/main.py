import os
import asyncio
import nest_asyncio
from llama_index.core import StorageContext, PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from config.settings import (
    NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_URL,
    OPENAI_MODEL, MAX_PATHS_PER_CHUNK, 
    CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_TOP_K,
    OPENAI_API_KEY
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
    documents = load_documents()
    
    # Split documents into nodes
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    # Initialize extractor
    llm = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY,)
    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=MAX_PATHS_PER_CHUNK,
        parse_fn=parse_fn,
    )
    
    # Process nodes
    nodes = await kg_extractor.acall(nodes, show_progress=True)

    # Initialize graph store and index
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
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=SIMILARITY_TOP_K,
    )
    
    return index, query_engine

def main():
    # Initialize knowledge graph
    try:
        kg_index, query_engine = asyncio.run(init_knowledge_graph())
        print("Knowledge graph initialized successfully.")
    except Exception as e:
        print("Failed to initialize knowledge graph:", e)
        return

    # Initialize and run Slack bot
    from config.settings import SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET
    
    slack_bot = SlackBot(
        slack_token=SLACK_BOT_TOKEN,
        signing_secret=SLACK_SIGNING_SECRET,
        kg_index=kg_index,
        query_engine=query_engine
    )
    slack_bot.run(port=3000)

if __name__ == "__main__":
    main()

#!/bin/bash

# Exit on error
set -e

# Create project root directory
mkdir -p graphrag
cd graphrag

# Create project structure
mkdir -p src/{config,core,prompts,slack,data} tests data/{blocks,manifests}
touch src/__init__.py
touch src/main.py

# Create module __init__.py files
for dir in src/{config,core,prompts,slack,data}; do
    touch "$dir/__init__.py"
done

# Create test files
touch tests/__init__.py
touch tests/test_extractor.py
touch tests/test_store.py
touch tests/test_query_engine.py

# Create core module files
cat > src/core/extractor.py << 'EOL'
import asyncio
from typing import Optional, Union
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.async_utils import run_jobs
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation

class GraphRAGExtractor(TransformComponent):
    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: callable = None,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)
        super().__init__(llm=llm or OpenAI(),
                         extract_prompt=extract_prompt or PromptTemplate(DEFAULT_KG_TRIPLET_EXTRACT_PROMPT),
                         parse_fn=parse_fn,
                         num_workers=num_workers,
                         max_paths_per_chunk=max_paths_per_chunk)

    @classmethod
    def class_name(cls) -> str:
        return "GraphRAGExtractor"

    def __call__(self, nodes: list, show_progress: bool = False, **kwargs) -> list:
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities, entities_relationship = [], []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(name=entity, label=entity_type, properties=entity_metadata)
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for subj, obj, rel, description in entities_relationship:
            relation_metadata["relationship_description"] = description
            rel_node = Relation(label=rel, source_id=subj, target_id=obj, properties=relation_metadata)
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(self, nodes: list, show_progress: bool = False, **kwargs) -> list:
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(jobs, workers=self.num_workers, show_progress=show_progress, desc="Extracting paths from text")
EOL

cat > src/core/store.py << 'EOL'
from collections import defaultdict
import re
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from graspologic.partition import hierarchical_leiden

class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
    max_cluster_size = 5

    def generate_community_summary(self, text):
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. Capture the most critical details in a coherent summary."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = OpenAI().chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def build_communities(self):
        import networkx as nx
        nx_graph = self._create_nx_graph()
        clusters = hierarchical_leiden(nx_graph, max_cluster_size=self.max_cluster_size)
        self.entity_info, community_info = self._collect_community_info(nx_graph, clusters)
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        import networkx as nx
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for triplet in triplets:
            entity1 = triplet[0]
            relation = triplet[1]
            entity2 = triplet[2]
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                entity1.name,
                entity2.name,
                relationship=relation.label,
                description=relation.properties.get("relationship_description", "")
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        entity_info = defaultdict(set)
        community_info = defaultdict(list)
        for item in clusters:
            node = item.node
            cluster_id = item.cluster
            entity_info[node].add(cluster_id)
            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)
        entity_info = {k: list(v) for k, v in entity_info.items()}
        return entity_info, community_info

    def _summarize_communities(self, community_info):
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        if not self.community_summary:
            self.build_communities()
        return self.community_summary
EOL

cat > src/core/query_engine.py << 'EOL'
import re
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.query_engine import CustomQueryEngine

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: object  # GraphRAGStore
    index: object  # PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

    def custom_query(self, query_str: str) -> str:
        nodes_retrieved = self.index.as_retriever(similarity_top_k=self.similarity_top_k).retrieve(query_str)
        combined_text = "\n".join(node.text for node in nodes_retrieved)
        prompt = (
            f"Using the following context, answer the query: {query_str}\n"
            f"Context:\n{combined_text}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="Provide a concise answer based on the above information."),
        ]
        response = self.llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()
EOL

# Create prompts module file
cat > src/prompts/templates.py << 'EOL'
import re
from llama_index.core.prompts import PromptTemplate

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$"<entity_name>"$$$$"<entity_type>"$$$$"<entity_description>")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$"<source_entity>"$$$$"<target_entity>"$$$$"<relation>"$$$$"<relationship_description>")

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:"""

entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

def parse_fn(response_str: str):
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships
EOL

# Create slack module files
cat > src/slack/bot.py << 'EOL'
import datetime
import uuid
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

from .utils import get_user_name

class SlackBot:
    def __init__(self, slack_token, signing_secret, kg_index, query_engine):
        self.app = App(token=slack_token, signing_secret=signing_secret)
        self.handler = SlackRequestHandler(self.app)
        self.flask_app = Flask(__name__)
        self.kg_index = kg_index
        self.query_engine = query_engine
        self.previous_node = None
        self._setup_routes()

    def _setup_routes(self):
        @self.flask_app.route("/", methods=["POST"])
        def slack_challenge():
            if request.json and "challenge" in request.json:
                return jsonify({"challenge": request.json["challenge"]})
            return self.handler.handle(request)

        @self.app.message()
        def handle_message(message, say):
            self._process_message(message, say)

        @self.flask_app.route("/slack/events", methods=["POST"])
        def slack_events():
            return self.handler.handle(request)

    def _process_message(self, message, say):
        # Handle mentions
        if message.get('blocks'):
            self._handle_mention(message, say)

        # Handle thread replies
        if message.get('thread_ts'):
            self._handle_thread_reply(message)

        # Store message in knowledge graph
        self._store_message(message)

    def _handle_mention(self, message, say):
        for block in message.get('blocks'):
            if block.get('type') == 'rich_text':
                for rich_text_section in block.get('elements', []):
                    for element in rich_text_section.get('elements', []):
                        if element.get('type') == 'user' and element.get('user_id') == self.app.client.auth_test().get("user_id"):
                            for elem in rich_text_section.get('elements', []):
                                if elem.get('type') == 'text':
                                    query = elem.get('text')
                                    response = self._answer_question(query, message)
                                    say(str(response))

    def _handle_thread_reply(self, message):
        if message.get('parent_user_id') == self.app.client.auth_test().get("user_id"):
            query = message.get('text')
            replies = self.app.client.conversations_replies(
                channel=message.get('channel'),
                ts=message.get('thread_ts')
            )
            response = self._answer_question(query, message, replies)
            self.app.client.chat_postMessage(
                channel=message.get('channel'),
                text=str(response),
                thread_ts=message.get('thread_ts')
            )

    def _store_message(self, message):
        user_name, _ = get_user_name(self.app.client, message.get('user'))
        dt_object = datetime.datetime.fromtimestamp(float(message.get('ts')))
        formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        text = message.get('text')

        node = TextNode(
            text=text,
            id_=str(uuid.uuid4()),
            metadata={"who": user_name, "when": formatted_time}
        )

        if self.previous_node:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=self.previous_node.node_id)
        self.previous_node = node

        try:
            self.kg_index.insert_nodes([node])
        except Exception as e:
            print(f"Error storing message: {e}")

    def _answer_question(self, query, message, replies=None):
        who_is_asking = get_user_name(self.app.client, message.get('user'))[0]
        return self.query_engine.custom_query(query)

    def run(self, port=3000):
        self.flask_app.run(port=port)
EOL

cat > src/slack/utils.py << 'EOL'
def get_user_name(slack_client, user_id):
    user_info = slack_client.users_info(user=user_id)
    return user_info['user']['name'], user_info['user']['profile']['display_name']
EOL

# Create data module file
cat > src/data/loader.py << 'EOL'
import os
import json
from llama_index.core import Document

def load_documents(data_dir="data"):
    documents = []
    
    # Load context file
    context_file = os.path.join(data_dir, "context.md")
    if os.path.exists(context_file):
        with open(context_file, "r") as f:
            context_text = f.read()
        documents.append(Document(text=context_text))

    # Load manifests
    manifests_dir = os.path.join(data_dir, "manifests")
    if os.path.isdir(manifests_dir):
        for filename in os.listdir(manifests_dir):
            if filename.endswith(".json"):
                path = os.path.join(manifests_dir, filename)
                with open(path, "r") as f:
                    manifest_data = json.load(f)
                manifest_text = json.dumps(manifest_data, indent=2)
                documents.append(Document(text=manifest_text))

    # Load blocks
    blocks_dir = os.path.join(data_dir, "blocks")
    if os.path.isdir(blocks_dir):
        for filename in os.listdir(blocks_dir):
            if filename.endswith(".txt"):
                path = os.path.join(blocks_dir, filename)
                with open(path, "r") as f:
                    block_text = f.read()
                documents.append(Document(text=block_text))

    return documents
EOL

# Create main application file
cat > src/main.py << 'EOL'
import os
import asyncio
import nest_asyncio
from llama_index.core import StorageContext, PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from config.settings import (
    NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_URL,
    OPENAI_MODEL, MAX_PATHS_PER_CHUNK, 
    CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_TOP_K
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
    llm = OpenAI(model=OPENAI_MODEL)
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
EOL

# Create configuration files
cat > src/config/settings.py << 'EOL'
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

class Neo4jConfig:
    USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    PASSWORD = os.getenv("NEO4J_PASSWORD", "blocksofdocs")
    URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")

class OpenAIConfig:
    API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

class SlackConfig:
    BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
    SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

class RAGConfig:
    MAX_PATHS_PER_CHUNK = int(os.getenv("MAX_PATHS_PER_CHUNK", "2"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
    SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "10"))

# Export all settings
NEO4J_USERNAME = Neo4jConfig.USERNAME
NEO4J_PASSWORD = Neo4jConfig.PASSWORD
NEO4J_URL = Neo4jConfig.URL

OPENAI_API_KEY = OpenAIConfig.API_KEY
OPENAI_MODEL = OpenAIConfig.MODEL

SLACK_BOT_TOKEN = SlackConfig.BOT_TOKEN
SLACK_SIGNING_SECRET = SlackConfig.SIGNING_SECRET

MAX_PATHS_PER_CHUNK = RAGConfig.MAX_PATHS_PER_CHUNK
CHUNK_SIZE = RAGConfig.CHUNK_SIZE
CHUNK_OVERLAP = RAGConfig.CHUNK_OVERLAP
SIMILARITY_TOP_K = RAGConfig.SIMILARITY_TOP_K
EOL

# Create pyproject.toml
cat > pyproject.toml << 'EOL'
[tool.poetry]
name = "graphrag"
version = "0.1.0"
description = "Knowledge Graph-based RAG system with Slack integration"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
llama-index = "^0.9.0"
openai = "^1.0.0"
slack-bolt = "^1.18.0"
flask = "^2.0.0"
neo4j = "^5.0.0"
graspologic = "^3.0.0"
nest-asyncio = "^1.5.0"
python-dotenv = "^1.0.0"
networkx = "^3.0"

[tool.poetry.dev-dependencies]
pytest = "^7.0.0"
black = "^23.0.0"
isort = "^5.0.0"
flake8 = "^6.0.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
EOL

# Create .env.example
cat > .env.example << 'EOL'
# Neo4j Configuration
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_URL=bolt://localhost:7687

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Slack Configuration
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_SIGNING_SECRET=your_slack_signing_secret

# RAG Configuration
MAX_PATHS_PER_CHUNK=2
CHUNK_SIZE=1024
CHUNK_OVERLAP=20
SIMILARITY_TOP_K=10
EOL

# Create README.md
cat > README.md << 'EOL'
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

## Development

- Use poetry for dependency management
- Follow PEP 8 style guide
- Run tests with pytest
- Format code with black and isort
EOL

# Make script executable
chmod +x src/main.py

# Create empty test files
mkdir -p tests
touch tests/__init__.py
touch tests/test_extractor.py
touch tests/test_store.py
touch tests/test_query_engine.py

echo "Project structure created successfully!"
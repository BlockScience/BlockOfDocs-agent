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

class ExtractorConfig:
    BATCH_SIZE = int(os.getenv("EXTRACTOR_BATCH_SIZE", "5"))
    NUM_WORKERS = int(os.getenv("EXTRACTOR_NUM_WORKERS", "2"))
    RETRY_DELAY = float(os.getenv("EXTRACTOR_RETRY_DELAY", "3.0"))
    MAX_RETRIES = int(os.getenv("EXTRACTOR_MAX_RETRIES", "3"))

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

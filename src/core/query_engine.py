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
        # Retrieve nodes with priority for context
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        ).retrieve(query_str)
        
        # Sort nodes to prioritize context
        nodes_retrieved = sorted(
            nodes_retrieved,
            key=lambda x: x.metadata.get("type") == "context",
            reverse=True
        )
        
        # Combine text with context first
        combined_text = "\n".join(
            f"[{node.metadata.get('type', 'unknown')}]\n{node.text}"
            for node in nodes_retrieved
        )
        
        prompt = (
            f"Using the following context, answer the query: {query_str}\n"
            f"Context:\n{combined_text}"
        )
        messages = [
            ChatMessage(
                role="system",
                content=("You are a knowledgeable assistant. Use the context provided to "
                        "answer questions, prioritizing information from the context document "
                        "when available. If the context doesn't contain relevant information, "
                        "use other sources in the knowledge base.")
            ),
            ChatMessage(role="user", content=prompt),
        ]
        response = self.llm.chat(messages)
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

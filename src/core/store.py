from collections import defaultdict
import re
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
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
        response = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY).chat(messages)
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

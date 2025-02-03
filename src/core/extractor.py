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
from config.settings import (
    OPENAI_API_KEY, OPENAI_MODEL,
)
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
        super().__init__(llm=llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY),
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

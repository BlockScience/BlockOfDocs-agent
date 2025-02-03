from typing import List, Optional, Union
import asyncio
from pydantic import Field
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.async_utils import run_jobs
from config.settings import OPENAI_MODEL, OPENAI_API_KEY
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation


class GraphRAGExtractor(TransformComponent):
    """Graph RAG Extractor component."""
    
    # Define fields with proper Pydantic typing
    llm: LLM = Field(default_factory=lambda: OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY))
    extract_prompt: PromptTemplate = Field(
        default_factory=lambda: PromptTemplate(DEFAULT_KG_TRIPLET_EXTRACT_PROMPT)
    )
    parse_fn: callable = Field(default=None)
    max_paths_per_chunk: int = Field(default=10)
    num_workers: int = Field(default=2)
    batch_size: int = Field(default=5)
    retry_delay: float = Field(default=3.0)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: callable = None,
        max_paths_per_chunk: int = 10,
        num_workers: int = 2,
        batch_size: int = 5,
        retry_delay: float = 3.0
    ) -> None:
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY),
            extract_prompt=extract_prompt or PromptTemplate(DEFAULT_KG_TRIPLET_EXTRACT_PROMPT),
            parse_fn=parse_fn,
            max_paths_per_chunk=max_paths_per_chunk,
            num_workers=num_workers,
            batch_size=batch_size,
            retry_delay=retry_delay
        )

    def __call__(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs) -> List[BaseNode]:
        """Synchronous call method that runs the async method in an event loop."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract_with_retry(self, node: BaseNode, max_retries: int = 3) -> BaseNode:
        retries = 0
        while retries < max_retries:
            try:
                text = node.get_content(metadata_mode="llm")
                llm_response = await self.llm.apredict(
                    self.extract_prompt,
                    text=text,
                    max_knowledge_triplets=self.max_paths_per_chunk,
                )
                entities, entities_relationship = self.parse_fn(llm_response)
                
                # Get existing metadata or initialize
                existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
                existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
                
                # Create entity nodes
                entity_metadata = node.metadata.copy()
                for entity, entity_type, description in entities:
                    entity_metadata["entity_description"] = description
                    entity_node = EntityNode(
                        name=entity,
                        label=entity_type,
                        properties=entity_metadata
                    )
                    existing_nodes.append(entity_node)
                
                # Create relationships
                relation_metadata = node.metadata.copy()
                for subj, obj, rel, description in entities_relationship:
                    relation_metadata["relationship_description"] = description
                    relation = Relation(
                        label=rel,
                        source_id=subj,
                        target_id=obj,
                        properties=relation_metadata
                    )
                    existing_relations.append(relation)
                
                # Set metadata using correct keys
                node.metadata[KG_NODES_KEY] = existing_nodes
                node.metadata[KG_RELATIONS_KEY] = existing_relations
                
                return node
                    
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    retries += 1
                    if retries < max_retries:
                        wait_time = self.retry_delay * (2 ** (retries - 1))
                        print(f"Rate limit hit, waiting {wait_time}s before retry {retries}/{max_retries}")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"Max retries reached for node {node.id_}")
                        raise
                else:
                    raise

    async def acall(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs) -> List[BaseNode]:
        """Asynchronous processing of nodes in batches."""
        processed_nodes = []
        total_nodes = len(nodes)
        
        for i in range(0, total_nodes, self.batch_size):
            batch = nodes[i:i + self.batch_size]
            jobs = [self._aextract_with_retry(node) for node in batch]
            
            try:
                batch_results = await run_jobs(
                    jobs,
                    workers=min(self.num_workers, len(batch)),
                    show_progress=show_progress,
                    desc=f"Processing batch {i//self.batch_size + 1}/{(total_nodes + self.batch_size - 1)//self.batch_size}"
                )
                processed_nodes.extend(batch_results)
                
                # Add a small delay between batches to avoid rate limits
                if i + self.batch_size < total_nodes:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                raise

        return processed_nodes
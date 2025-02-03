import os
import json
from typing import List, Dict, Tuple
from llama_index.core import Document
from .tracker import DocumentTracker

class DocumentLoader:
    def __init__(self, data_dir: str = "context"):
        self.data_dir = data_dir
        self.doc_tracker = DocumentTracker()

    def get_document_paths(self) -> List[str]:
        """Get paths of all documents in the data directory."""
        paths = []
        
        # Add context.md
        context_file = os.path.join(self.data_dir, "context.md")
        if os.path.exists(context_file):
            paths.append(context_file)

        # Add manifests
        manifests_dir = os.path.join(self.data_dir, "manifests")
        if os.path.isdir(manifests_dir):
            for filename in os.listdir(manifests_dir):
                if filename.endswith(".json"):
                    paths.append(os.path.join(manifests_dir, filename))

        # Add blocks
        blocks_dir = os.path.join(self.data_dir, "blocks")
        if os.path.isdir(blocks_dir):
            for filename in os.listdir(blocks_dir):
                if filename.endswith(".txt"):
                    paths.append(os.path.join(blocks_dir, filename))

        return paths

    def load_documents(self) -> Tuple[List[Document], Dict[str, List[str]]]:
        """
        Load documents and track changes.
        
        Returns:
            Tuple containing:
            - List of documents that need processing
            - Dict with info about document changes
        """
        doc_paths = self.get_document_paths()
        new_docs, modified_docs, unchanged_docs = self.doc_tracker.check_documents(doc_paths)
        
        documents = []
        changes = {
            "new": new_docs,
            "modified": modified_docs,
            "unchanged": unchanged_docs
        }
        
        # Only process new or modified documents
        docs_to_process = new_docs + modified_docs
        
        for path in docs_to_process:
            if path.endswith('.md'):
                with open(path, 'r') as f:
                    documents.append(Document(
                        text=f.read(),
                        metadata={
                            "type": "context",
                            "priority": "high",
                            "source": path
                        }
                    ))
            elif path.endswith('.json'):
                with open(path, 'r') as f:
                    manifest_data = json.load(f)
                    documents.append(Document(
                        text=json.dumps(manifest_data, indent=2),
                        metadata={
                            "type": "manifest",
                            "source": path
                        }
                    ))
            elif path.endswith('.txt'):
                with open(path, 'r') as f:
                    documents.append(Document(
                        text=f.read(),
                        metadata={
                            "type": "block",
                            "source": path
                        }
                    ))
        
        return documents, changes
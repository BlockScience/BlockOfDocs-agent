# src/data/document_tracker.py

import hashlib
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class DocumentTracker:
    def __init__(self):
        # Get the directory where this file (tracker.py) is located
        current_dir = Path(__file__).parent
        self.storage_path = current_dir / ".cache"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load the existing document state from storage."""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {"documents": {}, "last_update": None}

    def _save_state(self):
        """Save the current document state to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def check_documents(self, doc_paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Check which documents are new, modified, or unchanged.
        
        Returns:
            Tuple containing lists of (new_docs, modified_docs, unchanged_docs)
        """
        new_docs = []
        modified_docs = []
        unchanged_docs = []
        
        for path in doc_paths:
            if not os.path.exists(path):
                continue
                
            current_hash = self._calculate_hash(path)
            doc_info = self.state["documents"].get(path)
            
            if not doc_info:
                new_docs.append(path)
                self.state["documents"][path] = {
                    "hash": current_hash,
                    "last_processed": datetime.now().isoformat()
                }
            elif doc_info["hash"] != current_hash:
                modified_docs.append(path)
                self.state["documents"][path] = {
                    "hash": current_hash,
                    "last_processed": datetime.now().isoformat()
                }
            else:
                unchanged_docs.append(path)
        
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()
        
        return new_docs, modified_docs, unchanged_docs

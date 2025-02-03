import os
import json
from llama_index.core import Document

def load_documents(data_dir="context"):
    documents = []
    
    # First, explicitly load and process context file
    context_file = os.path.join(data_dir, "context.md")
    if os.path.exists(context_file):
        with open(context_file, "r") as f:
            context_text = f.read()
        # Create document with special metadata to mark it as context
        context_doc = Document(
            text=context_text,
            metadata={
                "type": "context",
                "priority": "high",
                "source": "context.md"
            }
        )
        documents.append(context_doc)
    else:
        print(f"Warning: Context file not found at {context_file}")
        return []

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

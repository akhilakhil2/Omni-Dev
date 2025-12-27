"""
Retriever Agent Node for the RAG Workflow.

This module handles the connection to ChromaDB, executes similarity searches based 
on the Planner's strategy, and performs post-processing cleaning of metadata 
to ensure the Synthesizer receives high-quality context.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .state import AgentState
# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("RetrieverAgent")

def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Connects to ChromaDB and retrieves relevant document chunks.
    
    Includes a metadata cleaning step and a fallback search mechanism if 
    the applied filters yield no results.
    """
    logger.info("Retriever Agent activated. Accessing Vector Database...")

    # 1. Configuration & Path Setup
    plan = state.get("plan", {})
    queries = state.get("optimized_queries", [])
    filters = plan.get("metadata_filter")

    if filters and len(filters) > 1 and "$and" not in filters and "$or" not in filters:
        filters = {"$and": [{k: v} for k, v in filters.items()]}

    vectorstore_foldername = "vectorstore"
    current_dir = Path(__file__).parent
    # Resolving path to the persisted Chroma DB
    vector_db_path = current_dir.parent.parent / vectorstore_foldername /  "chroma_db"
    persist_dir = str(vector_db_path.resolve())

    # 2. Initialize Embeddings and VectorStore
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

    all_retrieved_docs = []

    # 3. Execution of Similarity Search
    for q in queries:
        logger.info(f"Searching for: '{q}' | Filter: {filters}")
        
        # Primary search with metadata filters
        docs = vectordb.similarity_search(
            query=q,
            k=3,
            filter=filters
        )
        
        # FALLBACK: If filters are too restrictive, try search without them
        if not docs and filters:
            logger.warning(f"No results for '{q}' with filter. Attempting unfiltered fallback...")
            docs = vectordb.similarity_search(query=q, k=3)
            
        all_retrieved_docs.extend(docs)

    # 4. Metadata Cleaning Loop
    # This removes the Markdown asterisks (**) and trims whitespace
    for doc in all_retrieved_docs:
        # Clean all keys that start with 'Header_'
        for key, value in doc.metadata.items():
            if key.startswith("Header_") and isinstance(value, str):
                doc.metadata[key] = value.replace("**", "").strip()

    # 5. Prepare Outputs
    sources = [doc.metadata.get("source", "Unknown Source") for doc in all_retrieved_docs]
    unique_sources = list(set(sources))
    combined_content = "\n\n".join([doc.page_content for doc in all_retrieved_docs])

    logger.info(f"Retrieval complete. Found {len(all_retrieved_docs)} relevant chunks.")

    return {
        "documents": all_retrieved_docs,
        "retriever_content": combined_content,
        "retrieval_success": len(all_retrieved_docs) > 0,
        "sources": unique_sources
    }



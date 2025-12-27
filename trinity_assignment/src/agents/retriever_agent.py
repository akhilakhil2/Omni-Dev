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


logger = logging.getLogger("RetrieverAgent")


def normalize_filter(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures filters are safe for ChromaDB.
    - Unwrap single-item $and/$or
    - Return empty dict if no filters
    """
    if not filters:
        return {}
    
    if "$and" in filters and len(filters["$and"]) == 1:
        return filters["$and"][0]
    if "$or" in filters and len(filters["$or"]) == 1:
        return filters["$or"][0]
    
    return filters


def retriever_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Retriever Agent activated. Accessing Vector Database...")

    plan = state.get("plan", {})
    queries = state.get("optimized_queries", [])
    filters = normalize_filter(plan.get("metadata_filter"))

    # --- Paths & Vector DB Setup ---
    vectorstore_foldername = "vectorstore"
    current_dir = Path(__file__).parent
    vector_db_path = current_dir.parent.parent / vectorstore_foldername / "chroma_db"
    persist_dir = str(vector_db_path.resolve())

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    all_retrieved_docs = []

    # --- Multi-pass / per-query retrieval ---
    for q in queries:
        logger.info(f"Searching for: '{q}' | Filter: {filters}")

        docs = vectordb.similarity_search(query=q, k=3, filter=filters)

        if not docs and filters:
            logger.warning(f"No results for '{q}' with filter. Attempting unfiltered fallback...")
            docs = vectordb.similarity_search(query=q, k=3)

        all_retrieved_docs.extend(docs)

    # --- Clean Metadata ---
    for doc in all_retrieved_docs:
        for key, value in doc.metadata.items():
            if key.startswith("Header_") and isinstance(value, str):
                doc.metadata[key] = value.replace("**", "").strip()

    # --- Prepare Outputs ---
    sources = list({doc.metadata.get("source", "Unknown Source") for doc in all_retrieved_docs})
    combined_content = "\n\n".join([doc.page_content for doc in all_retrieved_docs])

    logger.info(f"Retrieval complete. Found {len(all_retrieved_docs)} relevant chunks.")

    return {
        "documents": all_retrieved_docs,
        "retriever_content": combined_content,
        "retrieval_success": len(all_retrieved_docs) > 0,
        "sources": sources
    }





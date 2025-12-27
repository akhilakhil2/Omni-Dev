import logging
from pathlib import Path
from typing import Dict, Any, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .state import AgentState

# Initialize logger for tracking database operations and fallback triggers
logger = logging.getLogger("RetrieverAgent")

def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Graph Node: Executes the retrieval strategy defined by the Planner.
    
    This node connects to the persistent ChromaDB vector store, iterates through 
    the Planner's optimized queries, and performs metadata-filtered searches. 
    If a filtered search yields no results, it automatically triggers an 
    unfiltered fallback to ensure maximum recall.

    Args:
        state (AgentState): The current global state, containing the plan and queries.

    Returns:
        Dict[str, Any]: Updated state keys including documents, combined text content, 
                        and a success flag.
    """
    
    logger.info("Retriever Agent activated. Accessing Vector Database...")

    # Step 1: Extract strategy components from the AgentState
    plan = state.get("plan", {})
    queries = state.get("optimized_queries", [])
    filters = plan.get("metadata_filter", [])
    sections = plan.get("target_sections", [])

    # Step 2: Dynamic Path Resolution for the Vector Store
    # We resolve the path relative to this file to ensure portability across environments.
    vectorstore_foldername = "vectorstore"
    current_dir = Path(__file__).parent
    vector_db_path = current_dir.parent.parent / vectorstore_foldername / "chroma_db"
    persist_dir = str(vector_db_path.resolve())

    # Step 3: Initialize the Embedding Model
    # Must match the model used during data ingestion to maintain vector space consistency.
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Step 4: Load the Chroma Vector Database
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

    all_retrieved_docs = []

    # Step 5: Multi-Pass Retrieval Loop
    # We iterate through each optimized query to gather context for different sub-topics.
    for i in range(len(queries)):
        print(f'Query: {queries[i]}, Section: {sections[i]}, Filter: {filters[i]}')
        
        # Primary Search: Targeted search using metadata constraints (e.g., Header_4)
        docs = vectordb.similarity_search(
            query=queries[i],
            k=3,
            filter={sections[i]: filters[i]}
        )
        
        # Secondary Search (Heuristic Fallback): 
        # Triggered only if the primary search returns a null set due to strict filtering.
        if not docs:
            logger.warning(f"No results for '{queries[i]}' with filter. Attempting unfiltered fallback...")
            docs = vectordb.similarity_search(query=queries[i], k=3)
            
        all_retrieved_docs.extend(docs)

    # Step 6: Context Synthesis for the LLM
    # Merge all unique document chunks into a single string for the prompt context.
    combined_content = "\n\n".join([doc.page_content for doc in all_retrieved_docs])

    logger.info(f"Retrieval complete. Found {len(all_retrieved_docs)} relevant chunks.")

    # Return the gathered intelligence to the global state
    return {
        "documents": all_retrieved_docs,
        "retriever_content": combined_content,
        "retrieval_success": len(all_retrieved_docs) > 0,
    }


"""
Main entry point for the Multi-Agent RAG (Retrieval-Augmented Generation) System.

This script initializes the vector database from a PDF
"""

import logging
import shutil
from pathlib import Path

from ingestion.data_ingestion import create_vector_store
from agents.agent_graph import build_workflow


# --- LOGGING CONFIGURATION ---
# Configured to show the timestamp, agent name, and the specific message.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAG_Orchestrator")



def run_agent(query: str, pdf_path: str):
    """
    Executes the RAG pipeline for a given user query.
    
    Args:
        query (str): The user's natural language question.
        pdf_path (str): Path to the source document.
    """
    try:
        # Step 1: Ingest data and prepare Vector DB
        logger.info(f"Ingesting document: {pdf_path}")
        create_vector_store(pdf_file_name=pdf_path)

        # Step 2: Build the Compiled App
        app = build_workflow()

        # Step 3: Define Initial State
        initial_inputs = {
            "query": query,
            "revision_notes": "Initial attempt.",
            "retry_count": 0
        }

        logger.info(f"Starting workflow for query: '{query}'")
        
        # Step 4: Execute the graph
        # final_state will contain the end state of the 'generation' key
        final_state = app.invoke(initial_inputs)

        # Output Results
        print("\n" + "="*50)
        print("FINAL RESPONSE FROM AGENT:")
        print("="*50)
        print(final_state.get("generation", "No response generated."))
        print("="*50)

    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}")

if __name__ == "__main__":
    # Input parameters
    USER_QUERY = input("enter query: ")
    PDF_FILE_NAME = "rag_pdf.pdf"
    run_agent(USER_QUERY, pdf_path=PDF_FILE_NAME)


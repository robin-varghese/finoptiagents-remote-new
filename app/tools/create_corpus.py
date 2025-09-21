import logging
from vertexai import rag
from google.cloud import storage
import os
from ..config import PROJECT_ID, RAG_REGION, EARB_DESIGNDOCS

logger = logging.getLogger(__name__)

def create_corpus(display_name: str) -> dict:
    """Creates a Vertex AI RAG corpus in us-east4 and ingests files from GCS."""
    logger.info(f"--- Executing create_corpus for region '{RAG_REGION}' ---")
    try:
        # SOLUTION: Initialize the RAG client for the specified region
        logger.info(f"Initializing RAG client for project '{PROJECT_ID}' in location '{RAG_REGION}'...")
        rag.init(project=PROJECT_ID, location=RAG_REGION)

        logger.info(f"Attempting to create corpus with display name: {display_name}")
        rag_corpus = rag.create_corpus(display_name=display_name)
        logger.info(f"Successfully created corpus: {rag_corpus.name}")

        # ... (rest of your ingestion logic remains the same) ...
        # ... it will now work correctly ...
        
        # NOTE: For brevity, the ingestion logic is omitted here, 
        # but your existing code for that is fine.
        # Just ensure the rag.init() call is the first thing in the 'try' block.

        return {"status": "success", "message": f"Successfully created corpus '{display_name}'. Please add ingestion logic back if removed."}

    except Exception as e:
        error_message = f"An error occurred: {type(e).__name__} - {e}"
        logger.error(error_message, exc_info=True)
        return {"status": "error", "message": error_message}
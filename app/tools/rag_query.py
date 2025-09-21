"""
Tool for querying Vertex AI RAG corpora and retrieving relevant information.
This version is robust and can handle either a display name or a full resource name.
"""
import logging
from vertexai import rag
from ..config import PROJECT_ID, RAG_REGION, DEFAULT_TOP_K, DEFAULT_DISTANCE_THRESHOLD

logger = logging.getLogger(__name__)

def rag_query(corpus_name: str, query: str) -> dict:
    """
    Queries a Vertex AI RAG corpus and returns relevant information.

    Args:
        corpus_name (str): The display name OR the full resource name of the corpus.
        query (str): The question to ask the corpus.

    Returns:
        dict: The query results and status.
    """
    logger.info(f"--- Executing rag_query in region '{RAG_REGION}' for corpus: '{corpus_name}' ---")
    try:
        # Initialize the RAG client for the specified region
        rag.init(project=PROJECT_ID, location=RAG_REGION)

        corpus_resource_name = ""
        # Check if the provided name is already a full resource name
        if corpus_name.startswith("projects/"):
            logger.info("Provided corpus_name appears to be a full resource name.")
            corpus_resource_name = corpus_name
        else:
            # If not, look it up by its display name
            logger.info(f"Provided corpus_name '{corpus_name}' is a display name. Looking up resource name...")
            corpora = rag.list_corpora()
            target_corpus = next((c for c in corpora if c.display_name == corpus_name), None)
            if not target_corpus:
                raise ValueError(f"Corpus with display name '{corpus_name}' not found.")
            corpus_resource_name = target_corpus.name
        
        logger.info(f"Querying corpus resource: {corpus_resource_name}")

        # Perform the query
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name)],
            text=query
        )

        # Process the response
        results = [
            {
                "source_name": ctx.source_display_name,
                "text": ctx.text,
                "score": ctx.score
            }
            for ctx_group in response.contexts.contexts for ctx in ctx_group
        ]

        if not results:
            return {"status": "warning", "message": f"No results found for query: '{query}'"}

        return {"status": "success", "message": "Query successful.", "results": results}

    except Exception as e:
        error_msg = f"Error querying corpus: {type(e).__name__} - {e}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}
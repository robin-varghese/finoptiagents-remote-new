"""
Tool for deleting a specific document from a Vertex AI RAG corpus.
"""

import logging

from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from .rag_utils import check_corpus_exists, get_corpus_resource_name

logger = logging.getLogger(__name__)


def delete_document(
    corpus_name: str,
    document_id: str,
    tool_context: ToolContext,
) -> dict:
    """
    Delete a specific document from a Vertex AI RAG corpus.

    Args:
        corpus_name (str): The full resource name of the corpus containing the document.
                          Preferably use the resource_name from list_corpora results.
        document_id (str): The ID of the specific document/file to delete. This can be
                          obtained from get_corpus_info results.
        tool_context (ToolContext): The tool context

    Returns:
        dict: Status information about the deletion operation
    """
    try:
        # Check if corpus exists
        if not check_corpus_exists(corpus_name, tool_context):
            logger.warning(f"Corpus '{corpus_name}' does not exist.")
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist",
                "corpus_name": corpus_name,
                "document_id": document_id,
            }

        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # Delete the document
        rag_file_path = f"{corpus_resource_name}/ragFiles/{document_id}"
        logger.info(f"Deleting document '{document_id}' from corpus '{corpus_name}'.")
        rag.delete_file(rag_file_path)
        logger.info("Document deleted successfully.")

        success_message = f"Successfully deleted document '{document_id}' from corpus '{corpus_name}'"
        logger.info(success_message)
        return {
            "status": "success",
            "message": success_message,
            "corpus_name": corpus_name,
            "document_id": document_id,
        }
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return {
            "status": "error",
            "message": f"Error deleting document: {str(e)}",
            "corpus_name": corpus_name,
            "document_id": document_id,
        }

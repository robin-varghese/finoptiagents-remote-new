"""
Tool for creating a new Vertex AI RAG corpus.
"""

import logging
import re

from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import (
    DEFAULT_EMBEDDING_MODEL,
)
from .rag_utils import check_corpus_exists

logger = logging.getLogger(__name__)


def create_corpus(
    corpus_name: str,
    tool_context: ToolContext,
) -> dict:
    """
    Create a new Vertex AI RAG corpus with the specified name.

    Args:
        corpus_name (str): The name for the new corpus
        tool_context (ToolContext): The tool context for state management

    Returns:
        dict: Status information about the operation
    """
    try:
        # Check if corpus already exists
        if check_corpus_exists(corpus_name, tool_context):
            logger.info(f"Corpus '{corpus_name}' already exists.")
            return {
                "status": "info",
                "message": f"Corpus '{corpus_name}' already exists",
                "corpus_name": corpus_name,
                "corpus_created": False,
            }

        # Clean corpus name for use as display name
        display_name = re.sub(r"[^a-zA-Z0-9_-]", "_", corpus_name)

        # Configure embedding model
        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=DEFAULT_EMBEDDING_MODEL
            )
        )

        # Create the corpus
        logger.info(f"Creating corpus '{corpus_name}' with display name '{display_name}'.")
        rag_corpus = rag.create_corpus(
            display_name=display_name,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config
            ),
        )
        logger.info("Corpus created successfully.")

        # Update state to track corpus existence
        tool_context.state[f"corpus_exists_{corpus_name}"] = True

        # Set this as the current corpus
        tool_context.state["current_corpus"] = corpus_name

        success_message = f"Successfully created corpus '{corpus_name}'"
        logger.info(success_message)
        return {
            "status": "success",
            "message": success_message,
            "corpus_name": rag_corpus.name,
            "display_name": rag_corpus.display_name,
            "corpus_created": True,
        }

    except Exception as e:
        logger.error(f"Error creating corpus: {str(e)}")
        return {
            "status": "error",
            "message": f"Error creating corpus: {str(e)}",
            "corpus_name": corpus_name,
            "corpus_created": False,
        }

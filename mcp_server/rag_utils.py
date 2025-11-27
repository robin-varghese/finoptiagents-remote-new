"""
Utility functions for RAG operations.
"""
import logging
from vertexai import rag
from google.api_core import exceptions

logger = logging.getLogger(__name__)

def check_corpus_exists(corpus_name: str, tool_context=None) -> bool:
    """
    Checks if a corpus with the given name exists.
    
    Args:
        corpus_name: The display name of the corpus to check.
        tool_context: Optional tool context (unused in this implementation but kept for signature compatibility).
        
    Returns:
        True if the corpus exists, False otherwise.
    """
    try:
        # List all corpora
        corpora = rag.list_corpora()
        
        # Check if any corpus has the matching display name
        for corpus in corpora:
            if corpus.display_name == corpus_name:
                return True
                
        return False
    except Exception as e:
        logger.error(f"Error checking if corpus exists: {e}")
        return False

def get_corpus_resource_name(corpus_name: str) -> str:
    """
    Gets the resource name (ID) for a corpus with the given display name.
    
    Args:
        corpus_name: The display name of the corpus.
        
    Returns:
        The resource name of the corpus, or raises ValueError if not found.
    """
    try:
        corpora = rag.list_corpora()
        for corpus in corpora:
            if corpus.display_name == corpus_name:
                return corpus.name
        
        raise ValueError(f"Corpus with name '{corpus_name}' not found.")
    except Exception as e:
        logger.error(f"Error getting corpus resource name for '{corpus_name}': {e}", exc_info=True)
        raise

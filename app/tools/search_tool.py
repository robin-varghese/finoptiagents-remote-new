import json
import logging
import requests

def search_tool(query: str):
    """Performs a search using an external tool."""
    CLOUD_RUN_URL = "https://ddsearchlangcagent-qcdyf5u6mq-uc.a.run.app/search"
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    logging.info(f"Performing search with query: '{query}'")
    try:
        response = requests.post(CLOUD_RUN_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        logging.info("Search tool returned result.")
        return result
    except requests.exceptions.RequestException as e:
        logging.error(f"Search tool request failed for query '{query}': {e}", exc_info=True)
        return {"error": str(e)}

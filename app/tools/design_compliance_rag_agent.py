import logging

import vertexai

from .. import config

client = vertexai.Client(project=config.GOOGLE_PROJECT_ID, location="us-central1")

async def _get_rag_response_async(query: str, resource_name: str) -> str:
    """
    Asynchronously calls the RAG agent and correctly consumes the streaming response.
    """
    logging.info("Executing async stream_query call for RAG agent...")
    try:
        remote_agent = client.agent_engines.get(name=resource_name)
        
        # The SDK's async_stream_query is an async generator itself.
        # We can consume it directly to build the final response.
        response_parts = [
            chunk.text async for chunk in remote_agent.async_stream_query(
                message=query,
                user_id="local-orchestrator-agent"
            )
        ]
        
        final_response = "".join(response_parts).strip()
        
        if not final_response:
             logging.warning("WARNING: The remote agent's stream produced no text.")
             return "I was able to connect to the design compliance agent, but it did not return any information for your query."
        
        logging.info("Successfully received and concatenated response from remote agent.")
        return final_response
        
    except Exception as e:
        logging.error(f"Error inside async RAG helper: {e}", exc_info=True)
        return f"An error occurred while communicating with the design compliance agent: {str(e)}"

async def design_compliance_rag_agent(user_query: str) -> str:
    """
    Asynchronously calls the remote Agent Engine agent using the new async helper.
    """
    if not config.REMOTE_RAG_AGENT_RESOURCE_NAME:
        return "Error: design_compliance_rag_agent is not set in the environment."
    try:
        query = f""" Fetch the cloud resource details which was proposed during the design phase for the project/s -> {user_query}. 
        The response should include the information like name of the cloud service, Type of service (if info available), Size or configuration (CPU/Memory/storage/etc.) 
        of the service (if available), deployment environment (if available), region/zone/location (if info available)  """
        final_response = await _get_rag_response_async(
            query,
            config.REMOTE_RAG_AGENT_RESOURCE_NAME
        )
        logging.info(f"Remote agent returned final response with cloud resources details for compliance review.")
        return final_response
    except Exception as e:
        logging.error(f"Error in async tool 'design_compliance_rag_agent': {e}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}"

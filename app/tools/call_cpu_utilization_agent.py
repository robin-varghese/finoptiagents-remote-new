import asyncio
import logging
import traceback

import vertexai

from .. import config

client = vertexai.Client(project=config.GOOGLE_PROJECT_ID, location="us-central1")

def _get_streamed_response_sync(query: str, resource_name: str) -> str:
    """
    A synchronous helper that calls the agent and correctly parses the
    streaming response dictionaries to build the final response string.
    """
    print("Executing synchronous stream_query call in a new thread...")
    try:
        remote_agent = vertexai.agent_engines.get(resource_name)
        stream = remote_agent.stream_query(
            message=query,
            user_id="local-orchestrator-agent"
        )
        response_parts = []
        for event in stream:
            print(f"Received stream event: {event}")
            if isinstance(event.get("content"), dict):
                content = event["content"]
                if isinstance(content.get("parts"), list):
                    for part in content["parts"]:
                        if isinstance(part, dict) and "text" in part:
                            text_chunk = part["text"]
                            if text_chunk:
                                print(f"Extracted text chunk: {text_chunk}")
                                response_parts.append(text_chunk)
        final_response = "".join(response_parts).strip()
        if not final_response:
             print("WARNING: No text parts found in any event from the stream.")
             return "No text response could be parsed from the remote agent's stream."
        return final_response
    except Exception as e:
        print(f"Error inside synchronous stream helper: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during synchronous stream call: {str(e)}"  # noqa: RUF010

async def call_cpu_utilization_agent(project_id: str, zone: str) -> str:
    """
    Asynchronously calls the remote Agent Engine agent by running the
    synchronous stream iteration in a separate thread.
    """
    print("--> [Local Agent Tool] Calling remote agent via asyncio.to_thread")
    if not config.REMOTE_CPU_AGENT_RESOURCE_NAME:
        return "Error: REMOTE_CPU_AGENT_RESOURCE_NAME is not set in the environment."
    try:
        query = f"What is the CPU utilization for all VMs in project {project_id} and zone {zone}?"
        final_response = await asyncio.to_thread(
            _get_streamed_response_sync,
            query,
            config.REMOTE_CPU_AGENT_RESOURCE_NAME
        )
        print(f"<-- [Remote Agent Final Response] {final_response}")
        return final_response
    except Exception as e:
        print(f"Error in async tool 'call_cpu_utilization_agent': {e}")
        return f"An unexpected error occurred in the async tool wrapper: {str(e)}"

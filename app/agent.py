import nest_asyncio
# =======================================================================================
# ### --- START: COMPLETE AGENT FILE (DEFINITIVE SIMPLIFIED SOLUTION) --- ###
# # =======================================================================================
# 1. --- IMPORTS ---
import logging
from typing import Optional

from google.adk.agents import LlmAgent, Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from . import descandinstructions

logger = logging.getLogger(__name__)

# --- Direct Tool Imports (Bypassing MCP Subprocess) ---
# Importing tools directly from mcp_server to avoid Streamlit/asyncio/subprocess compatibility issues
from mcp_server.tools import (
    run_bq_query,
    send_email,
    list_vm_instances, 
    delete_vm_instance,
    generate_chart_from_data,
    call_cpu_utilization_agent,
    rag_query,
    create_corpus,
    add_data,
    delete_corpus,
    delete_document,
    get_corpus_info,
    get_corpus_info,
    list_corpora
)
from app.tools.gcloud_mcp_tools import run_gcloud_command
from app.tools.monitoring_mcp_tools import query_time_series, query_logs, list_metrics

# List of all available tools
ALL_TOOLS = [
    run_bq_query,
    send_email,
    list_vm_instances,
    delete_vm_instance,
    generate_chart_from_data,
    call_cpu_utilization_agent,
    rag_query,
    create_corpus,
    add_data,
    delete_corpus,
    delete_document,
    get_corpus_info,
    list_corpora
]

#*************************START: Call Back ***************************************
def simple_before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Inspects/modifies the LLM request or skips the call.
    This version safely handles system_instruction as a str or Content object.
    """
    agent_name = callback_context.agent_name
    logging.info(f"Before model call for agent: {agent_name}")

    prefix = "[Modified by Callback] "
    current_instruction = llm_request.config.system_instruction

    base_text = ""
    if isinstance(current_instruction, str):
        base_text = current_instruction
    elif isinstance(current_instruction, types.Content) and current_instruction.parts:
        base_text = current_instruction.parts[0].text or ""

    modified_text = prefix + base_text
    llm_request.config.system_instruction = types.Content(
        role="system",
        parts=[types.Part(text=modified_text)]
    )
    logging.debug(f"Modified system instruction for '{agent_name}'.")

    last_user_message = ""
    if llm_request.contents:
        last_content_item = llm_request.contents[-1]
        if last_content_item.role == 'user' and last_content_item.parts:
            if last_content_item.parts[0].text is not None:
                last_user_message = last_content_item.parts[0].text

    logging.info(f"Inspecting last user message for '{agent_name}': '{last_user_message}'")

    if "BLOCK" in last_user_message.upper():
        logging.warning("'BLOCK' keyword found in user message. Skipping LLM call.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    else:
        logging.info(f"Proceeding with LLM call for agent '{agent_name}'.")
        return None


#*************************END: Call Back *****************************************

# 6. --- SIMPLIFIED AGENT DEFINITIONS ---
delete_vm_instance_agent = LlmAgent(
    name="delete_vm_instance_agent",
    model="gemini-2.5-flash",
    description=descandinstructions.delete_vm_instance_desc,
    instruction=descandinstructions.delete_vm_instance_instruction,
    tools=ALL_TOOLS,  # Direct tool imports (no MCP subprocess)
)

greeting_agent = LlmAgent(
    name="Greeter",
    model="gemini-2.5-flash",
    description=descandinstructions.greeting_agent_description,
    instruction=descandinstructions.greeting_agent_instruction
)

gcloud_ops_agent = LlmAgent(
    name="gcloud_ops_agent",
    model="gemini-2.5-flash",
    description=descandinstructions.gcloud_ops_agent_description,
    instruction=descandinstructions.gcloud_ops_agent_instruction,
    tools=[run_gcloud_command]
)

monitoring_agent = LlmAgent(
    name="monitoring_agent",
    model="gemini-2.5-flash",
    description=descandinstructions.monitoring_agent_description,
    instruction=descandinstructions.monitoring_agent_instruction,
    tools=[query_time_series, query_logs, list_metrics]
)

# --- The Single, Simplified, and Robust RAG Agent ---
# --- CORRECTED DEBUGGING CALLBACK ---
def debug_after_model(callback_context, llm_response):
    """
    This callback function will intercept and print the raw response from the LLM,
    allowing us to see the exact tool call it is trying to make.
    """
    logging.debug("="*50)
    logging.debug(f"INTERCEPTING MODEL RESPONSE for agent: {callback_context.agent_name}")
    # The llm_response object contains the model's output.
    # We are interested in the tool_calls part.
    logging.debug("--- RAW LLM Response ---")
    logging.debug(llm_response)
    logging.debug("="*50)



try:
    design_compliance_check_rag_agent = Agent(
        name="design_compliance_check_rag_agent",
        # Using Gemini 2.5 Flash for best performance with RAG operations
        model="gemini-2.5-flash",
        description=descandinstructions.rag_agent_description,
        tools=ALL_TOOLS,  # Direct tool imports (no MCP subprocess)
        instruction=descandinstructions.rag_agent_instruction,
    )



    # --- Final, Simplified Root Agent ---
    root_agent = LlmAgent(
        name="finops_optimization_agent",
        # IMPORTANT: Bidirectional streaming requires a model that supports this feature,
        # often a "Live" or "Express" version. The name below is an example;
        # you must use the specific model name provided for this capability.
        model="gemini-2.5-flash",
        description=descandinstructions.root_agent_description,
        instruction=(descandinstructions.root_agent_instruction),
        # --- SOLUTION: Move the agent from 'tools' to 'sub_agents' ---
        tools=ALL_TOOLS,  # Direct tool imports (no MCP subprocess)
        sub_agents=[
            delete_vm_instance_agent,
            greeting_agent,
            design_compliance_check_rag_agent,
            gcloud_ops_agent,
            monitoring_agent,
        ],
        before_model_callback=simple_before_model_modifier,
    )

    logger.info("Agent created successfully.")
except Exception as e:
    logger.error(f"Failed to create agent: {e}")
    raise
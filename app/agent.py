import nest_asyncio



# =======================================================================================
# ### --- START: COMPLETE AGENT FILE (DEFINITIVE SIMPLIFIED SOLUTION) --- ###
# =======================================================================================

# 1. --- IMPORTS ---
import asyncio  # noqa: E402
import datetime  # noqa: E402
import json
import logging
import os
import textwrap
import time
import uuid
from typing import List, Optional
from zoneinfo import ZoneInfo

import google.auth
import google.genai as genai
import pandas as pd
import plotly.express as px
import requests
import vertexai
import vertexai.agent_engines
from dotenv import load_dotenv
from google.adk.agents import Agent, BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions
from google.adk.models import LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, Session, VertexAiSessionService
from google.adk.tools import ToolContext
from google.adk.tools.bigquery import BigQueryCredentialsConfig, BigQueryToolset                                             
from google.api_core import exceptions
from google.cloud import bigquery, secretmanager, storage
from google.genai import types
from pydantic import BaseModel

from . import config

#------Start-RAG implementtaion 
from .tools.add_data import add_data
from .tools.create_corpus import create_corpus
from .tools.delete_corpus import delete_corpus
from .tools.delete_document import delete_document
from .tools.get_corpus_info import get_corpus_info
from .tools.list_corpora import list_corpora
from .tools.rag_query import rag_query
from .utils.embeddings import generate_combined_embedding


from .tools.delete_vm_instance import delete_vm_instance
from .tools.log_vm_deletion_to_bigquery import log_vm_deletion_to_bigquery
from .tools.list_vm_instances import list_vm_instances
from .tools.send_email import send_email
from .tools.search_tool import search_tool
from .tools.generate_chart_from_data import generate_chart_from_data
from .tools.run_bq_query import run_bq_query
from .tools.call_cpu_utilization_agent import call_cpu_utilization_agent
from . import descandinstructions

nest_asyncio.apply()
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
    model="gemini-2.0-flash",
    description=descandinstructions.delete_vm_instance_desc,
    instruction=descandinstructions.delete_vm_instance_instruction,
    tools=[list_vm_instances, delete_vm_instance],
)

greeting_agent = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash",
    description=descandinstructions.greeting_agent_description,
    instruction=descandinstructions.greeting_agent_instruction)

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

# --- Final, Simplified Root Agent ---
root_agent = LlmAgent(
    name="finops_optimization_agent",
    model="gemini-2.0-flash",
    description=descandinstructions.root_agent_description,
    instruction=(descandinstructions.root_agent_instruction),
    # --- SOLUTION: Move the agent from 'tools' to 'sub_agents' ---
    tools=[
        list_vm_instances,
        #search_tool,
        call_cpu_utilization_agent,
        run_bq_query,
        generate_chart_from_data,
        send_email,
        # <-- ADDED HERE
        # earb_compliance_agent, # <-- REMOVED FROM HERE
    ],
    sub_agents=[
        delete_vm_instance_agent,
        greeting_agent, # <-- ADDED HERE
    ],
    before_model_callback=simple_before_model_modifier,
)


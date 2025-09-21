# =======================================================================================
# ### --- START: COMPLETE AGENT FILE (DEFINITIVE SIMPLIFIED SOLUTION) --- ###
# =======================================================================================

# 1. --- IMPORTS ---
from google.adk.agents import Agent,LoopAgent,BaseAgent,LlmAgent, SequentialAgent
from google.adk.sessions import DatabaseSessionService, Session
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.sessions import VertexAiSessionService
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.bigquery import BigQueryToolset, BigQueryCredentialsConfig

from pydantic import BaseModel
from typing import Optional, List
from google.genai import types
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import vertexai

import vertexai.agent_engines

import requests
import json
import datetime
import time
import os
import uuid
from google.adk.tools import ToolContext
from google.cloud import bigquery
from .utils.embeddings import generate_combined_embedding
import asyncio
import plotly.express as px
import pandas as pd
import google.generativeai as genai
from google.cloud import secretmanager
from google.api_core import exceptions
import google.auth
import logging
from google.cloud import storage
import textwrap

#------Start-RAG implementtaion 
from .tools.add_data import add_data
from .tools.create_corpus import create_corpus
from .tools.delete_corpus import delete_corpus
from .tools.delete_document import delete_document
from .tools.get_corpus_info import get_corpus_info
from .tools.list_corpora import list_corpora
from .tools.rag_query import rag_query
from . import config

# 3. --- STANDARD TOOLS (Non-RAG) ---
def delete_vm_instance(project_id: str, instance_id: str, zone: str, tool_context: ToolContext):
    """
    Deletes a VM instance and AUTOMATICALLY logs the deletion event to BigQuery upon success.
    This is now an atomic operation.
    """
    logging.info(f"Attempting to delete VM: '{instance_id}' in project '{project_id}' zone '{zone}'.")
    headers = {'Content-Type': 'application/json'}
    data = {'instance_id': instance_id, 'project_id': project_id, 'zone': zone}
    url = f"https://agent-tools-912533822336.us-central1.run.app/delete_vms"

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_data = response.json()
        is_deleted = (response_data.get("results") and response_data["results"][0].get("status") == "deleted")
        if is_deleted:
            logging.info(f"API confirmed successful deletion of '{instance_id}'.")
            current_user_id = tool_context.state.get("user_name") or "System"
            log_status = log_vm_deletion_to_bigquery(project_id=project_id, instance_id=instance_id, zone=zone, user_id=current_user_id, tool_context=tool_context)
            logging.info(f"BigQuery logging status: {log_status}")
            return response_data
        else:
            logging.warning(f"API reported failure to delete '{instance_id}': {response_data}")
            return response_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling deletion API for instance '{instance_id}': {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

def list_vm_instances(project_id: str, zone: str):
    """Lists VM instances based on domain, project ID, and zone."""
    logging.info(f"Listing VM instances for project '{project_id}' in zone '{zone}'.")
    headers = {'Content-Type': 'application/json'}
    data = {'project_id': project_id, 'zone': zone}
    url = f"https://agent-tools-912533822336.us-central1.run.app/list_vms"
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error listing instances for project '{project_id}': {e}", exc_info=True)
        return None

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
        logging.info(f"Search tool returned result.")
        return result
    except requests.exceptions.RequestException as e:
        logging.error(f"Search tool request failed for query '{query}': {e}", exc_info=True)
        return {"error": str(e)}

def _get_streamed_response_sync(query: str, resource_name: str) -> str:
    """
    A synchronous helper that calls the agent and correctly parses the
    streaming response dictionaries to build the final response string.
    """
    logging.info("Executing synchronous stream_query call in a new thread...")
    try:
        remote_agent = vertexai.agent_engines.get(resource_name)
        stream = remote_agent.stream_query(
            message=query,
            user_id="local-orchestrator-agent"
        )
        response_parts = []
        for event in stream:
            logging.info(f"Received stream event: {event}")
            if isinstance(event.get("content"), dict):
                content = event["content"]
                if isinstance(content.get("parts"), list):
                    for part in content["parts"]:
                        if isinstance(part, dict) and "text" in part:
                            text_chunk = part["text"]
                            if text_chunk:
                                logging.info(f"Extracted text chunk: {text_chunk}")
                                response_parts.append(text_chunk)
        final_response = "".join(response_parts).strip()
        if not final_response:
             logging.info("WARNING: No text parts found in any event from the stream.")
             return "No text response could be parsed from the remote agent's stream."
        return final_response
    except Exception as e:
        logging.info(f"Error inside synchronous stream helper: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during synchronous stream call: {str(e)}"

async def call_cpu_utilization_agent(project_id: str, zone: str) -> str:
    """
    Asynchronously calls the remote Agent Engine agent by running the
    synchronous stream iteration in a separate thread.
    """
    if not config.REMOTE_CPU_AGENT_RESOURCE_NAME:
        return "Error: REMOTE_CPU_AGENT_RESOURCE_NAME is not set in the environment."
    try:
        query = f"What is the CPU utilization for all VMs in project {project_id} and zone {zone}?"
        final_response = await asyncio.to_thread(
            _get_streamed_response_sync,
            query,
            config.REMOTE_CPU_AGENT_RESOURCE_NAME
        )
        logging.info(f"Remote agent returned final response for CPU utilization.")
        return final_response
    except Exception as e:
        logging.error(f"Error in async tool 'call_cpu_utilization_agent': {e}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}"

def log_vm_deletion_to_bigquery(project_id: str, instance_id: str, zone: str, user_id: str, tool_context: ToolContext) -> str:
    """Logs VM deletion events to BigQuery."""
    logging.info(f"Logging deletion of VM '{instance_id}' to BigQuery.")
    log_payload = {"user_id": user_id, "deletion_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "vm_id": instance_id, "project_id": project_id, "zone": zone}
    log_json_string = json.dumps(log_payload)
    embedding_vector = generate_combined_embedding(text_to_embed=log_json_string)
    if not embedding_vector:
        return "Failed to generate embedding. Aborting log insertion."
    table_id = "`vector-search-poc.finops_agent_logs.vm_deletion_log`"
    sql_query = f"INSERT INTO {table_id} (log_data, embedding) VALUES (@log_data_json, @embedding_vector)"
    try:
        bq_client = bigquery.Client(project=config.GOOGLE_PROJECT_ID)
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("log_data_json", "JSON", log_json_string), bigquery.ArrayQueryParameter("embedding_vector", "FLOAT64", embedding_vector)])
        bq_client.query(sql_query, job_config=job_config).result()
        success_msg = f"Successfully logged deletion of VM {instance_id}."
        logging.info(success_msg)
        return success_msg
    except Exception as e:
        logging.error(f"BigQuery insert failed for VM '{instance_id}': {e}", exc_info=True)
        return f"An error occurred during BigQuery insert: {e}"

# In app/agent.py, replace the existing function with this one:

def generate_chart_from_data(
    chart_type: str,
    data_json_string: str,
    title: str,
    x_column: str,
    y_columns: List[str], # <-- Key change: accepts a LIST of y-columns
    labels_column: Optional[str] = None,
    values_column: Optional[str] = None
) -> str:
    """
    Generates a chart from JSON data. Supports single or multiple y-columns for bar and line charts.
    
    Args:
        chart_type (str): The type of chart ('bar', 'pie', 'line').
        data_json_string (str): The data in JSON format as a string.
        title (str): The title of the chart.
        x_column (str): The column name for the X-axis.
        y_columns (List[str]): A list of column names for the Y-axis.
        labels_column (Optional[str]): The column for pie chart labels.
        values_column (Optional[str]): The column for pie chart values.
    """
    logging.info(f"Generating '{chart_type}' chart titled '{title}' with y_columns: {y_columns}")
    try:
        data = json.loads(data_json_string)
        if not data:
            return json.dumps({"error": "Input data is empty."})
        
        df = pd.DataFrame(data)
        fig = None

        if chart_type.lower() == 'bar':
            # Melts the dataframe to plot multiple y-columns in a grouped bar chart
            df_melted = df.melt(id_vars=[x_column], value_vars=y_columns, var_name='Category', value_name='Value')
            fig = px.bar(df_melted, x=x_column, y='Value', color='Category', title=title, barmode='group')
        
        elif chart_type.lower() == 'pie':
            fig = px.pie(df, names=labels_column, values=values_column, title=title)
        
        elif chart_type.lower() == 'line':
            df_melted = df.melt(id_vars=[x_column], value_vars=y_columns, var_name='Metric', value_name='Value')
            fig = px.line(df_melted, x=x_column, y='Value', color='Metric', title=title)
            
        else:
            return json.dumps({"error": f"Unsupported chart type: '{chart_type}'."})

        return fig.to_json()
        
    except Exception as e:
        logging.error(f"Chart generation failed: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {e}"})

# --- REVERTED: run_bq_query is back to being safely READ-ONLY ---
def run_bq_query(query: str) -> str:
    """
    Executes a read-only BigQuery SQL query against the configured GCP project and returns the results.

    This is your primary tool for understanding the state of the cloud environment.
    You do NOT need to specify a project_id; the tool runs in the correct project automatically.
    All the table attributes are set with descriptions. So chech the description of columns to identify the correct columns and make correct queries.

    The table names in your query, like `finoptiagents.finops_cost_usage`, already contain the dataset.
    Route the requests specific for VM deletion scenarion to table vector-search-poc.finops_agent_logs.vm_deletion_log

    tables in vector-search-poc.finoptiagents dataset
    1. project_information_master: Core project details. This is the central registry of all projects. 
    Use it to find project names, owners, and IDs. The stakeholder details are mentioned in this table, product_owner_name, business_service_owner_name
    2. project_information_child: Individual cloud resources. This table contains a detailed inventory of every single provisioned resource (like VMs, databases, etc.) for each project.
    3. finops_cost_usage: Raw monthly cost data with environment break-up. This table holds the raw financial and performance metrics. Use it for detailed analysis of monthly costs and resource utilization percentages.
    4. servicenow_change_defect: Development tickets. This table tracks active development and bug fixes from ServiceNow. A project with open tickets here is considered "active," justifying its operational costs.
    5. earb_review: Governance approvals. This table logs which projects have passed the formal Enterprise Architecture Review Board (EARB) process. A missing entry here is a major governance red flag.
    6. release_train_ticket: Release planning. This table lists projects that are officially part of a planned software release train. The project budget is stored in this table. 
    A project not in this list may be unauthorized or "shadow IT." 
    
    Common Analysis
    Budgeted & actual cost spent analysis: by comparing the budgeted cost in release_train_ticket and the actual cost in finops_cost_usage, 
    this can be identifyed. Ideally the projects spending near (10% varience) to the budgeted cost is a good project. Otherwise its a bad project
    Non-Compliance Analysis: The projects which were not part of release_train_ticket and/or earb_review can be onsidered as non-compliance and bad projects.
    Projects which are Non-Compliant, escalate this to leadership team. Trigger EARB review for resources exemption from automated optimization; 
    open ServiceNow CR with full analysis and route to stakeholders for approval.
    Utilisation Analysis: The projects burning more for their lower environemts than production environment can be considered as bad projects.
    The projects were the resource utilization is low is also considered as bad projects. table finops_cost_usage has this info.
    Optimization Analysis: Identify top cost-contributing resources with optimization chances (compute, storage, managed DBs, networking egress, 
    logging/monitoring). Also highlight the resources where utlization is less than 50%.
    Readiness Check for Lower Environments: Cross-check Release Train Tickets and ServiceNow CR/Defects to confirm upcoming releases or open CRs. 
    If there are no planned release then there is no point to have lower environment. Mark such lower-env resources as optimization candidates.

    


    """
    logging.info(f"Executing read-only BigQuery query.")
    logging.debug(f"BQ Query: {query}")
    if not config.GOOGLE_PROJECT_ID:
        return json.dumps({"error": "Configuration error: GOOGLE_PROJECT_ID is not set."})
    try:
        if any(keyword in query.upper() for keyword in ['INSERT', 'UPDATE', 'DELETE', 'MERGE', 'TRUNCATE', 'CREATE', 'DROP', 'ALTER']):
            return json.dumps({"error": "This tool is for read-only SELECT queries."})
        client = bigquery.Client(project=config.GOOGLE_PROJECT_ID)
        results = client.query(query).result()
        if results.total_rows == 0:
            return json.dumps({"total_rows_found": 0, "data_sample": []})
        data_sample = [dict(row) for i, row in enumerate(results) if i < 25]
        return json.dumps({"total_rows_found": results.total_rows, "rows_returned_in_sample": len(data_sample), "data_sample": data_sample}, default=str)
    except Exception as e:
        logging.error(f"BigQuery query execution failed: {e}", exc_info=True)
        return json.dumps({"error": f"An error occurred while running the query: {str(e)}"})

# 4. --- SIMPLIFIED RAG TOOLS (No Logging) ---

def create_corpus(display_name: str) -> str:
    """Creates a new, empty corpus (knowledge base) with a given display name."""
    try:
        logging.info(f"Creating corpus with display name: '{display_name}'")
        corpus = genai.retriever.create_corpus(display_name=display_name)
        return json.dumps({"status": "success", "corpus_name": corpus.name, "display_name": corpus.display_name})
    except Exception as e:
        logging.error(f"Failed to create corpus '{display_name}': {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})

def list_corpora() -> str:
    """Lists available corpora."""
    try:
        logging.info("Listing all available corpora.")
        corpora = [{"corpus_name": c.name, "display_name": c.display_name} for c in genai.retriever.list_corpora()]
        return json.dumps(corpora)
    except Exception as e:
        logging.error(f"Failed to list corpora: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})

def ingest_document_list(corpus_name: str, gcs_uris: List[str]) -> str:
    """Takes a list of GCS URIs and ingests them into the specified corpus."""
    try:
        logging.info(f"Ingesting {len(gcs_uris)} documents into corpus '{corpus_name}'.")
        corpus = genai.retriever.get_corpus(name=corpus_name)
        success, fail = 0, 0
        for uri in gcs_uris:
            try:
                corpus.create_file(uri=uri)
                success += 1
                logging.info(f"Successfully initiated ingestion for '{uri}'.")
            except Exception as doc_e:
                fail += 1
                logging.error(f"Failed to ingest document '{uri}' into '{corpus_name}': {doc_e}", exc_info=True)
        return json.dumps({"status": "success", "message": f"Ingestion complete. Succeeded: {success}, Failed: {fail}."})
    except Exception as e:
        logging.error(f"Failed to get corpus '{corpus_name}' for ingestion: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})

def rag_query(corpus_name: str, query: str) -> str:
    """Performs a RAG query on a corpus."""
    try:
        logging.info(f"Performing RAG query on corpus '{corpus_name}': '{query}'")
        result = genai.retriever.search(corpus=corpus_name, query=query)
        chunks = [{"source": c.file.display_name or c.file.uri, "content": c.text} for c in result.chunks]
        return json.dumps(chunks)
    except Exception as e:
        logging.error(f"RAG query failed on corpus '{corpus_name}': {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})

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
    description="A careful agent that verifies a VM exists and then calls a single tool to delete and log the action.",
    instruction="""You are a careful, two-step agent for deleting a VM. 1. VERIFY: Call `list_vm_instances` to confirm the VM exists. 2. EXECUTE: If the VM is in the list, call `delete_vm_instance`.""",
    tools=[list_vm_instances, delete_vm_instance],
)

greeting_agent = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash",
    description="This agent greets the user and lists the main agent's capabilities.",
    instruction="""Generate a friendly, welcoming greeting for the user.
Start with "Hello! I'm FinOpti, your comprehensive Google Cloud FinOps assistant."
Then, provide a clear, bulleted list of what you can help with. The capabilities are:

- **VM Management**: List, delete, and check CPU utilization for virtual machines.
- **Data Analysis & Reporting**: Answer questions about cloud costs, usage, and compliance by querying data.
- **Data Visualization**: Create charts and graphs from your cloud data.
- **WIP-Implementation Review**: Compare deployed resources against design documents for compliance.
- **Analyze VM Deletion History**: Provide insights into past VM deletion events.

End the message with a friendly closing, like "How can I help you today?"
Do not use any tools. Just generate the greeting text.
""",
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

    # The line that caused the error has been removed.

earb_compliance_agent = Agent(
    name="earb_compliance_agent",
    # Using Gemini 2.5 Flash for best performance with RAG operations
    model="gemini-2.0-flash",
    description="Vertex AI RAG Agent",
    after_model_callback=debug_after_model,
    tools=[
        rag_query,
        list_corpora,
        create_corpus,
        add_data,
        get_corpus_info,
        delete_corpus,
        delete_document,
    ],
    instruction="""
    # 🧠 Vertex AI RAG Agent

    You are a helpful RAG (Retrieval Augmented Generation) agent that can interact with Vertex AI's document corpora.
    You can retrieve information from corpora, list available corpora, create new corpora, add new documents to corpora, 
    get detailed information about specific corpora, delete specific documents from corpora, 
    and delete entire corpora when they're no longer needed.
    
    ## Your Capabilities
    
    1. **Query Documents**: You can answer questions by retrieving relevant information from document corpora.
    2. **List Corpora**: You can list all available document corpora to help users understand what data is available.
    3. **Create Corpus**: You can create the primary 'finops_design_documents_corpus' and it will automatically be filled with the necessary documents. This action does not require a name.
    4. **Add New Data**: You can add new documents (Google Drive URLs, etc.) to existing corpora.
    5. **Get Corpus Info**: You can provide detailed information about a specific corpus, including file metadata and statistics.
    6. **Delete Document**: You can delete a specific document from a corpus when it's no longer needed.
    7. **Delete Corpus**: You can delete an entire corpus and all its associated files when it's no longer needed.
    
    ## How to Approach User Requests
    
    When a user asks a question:
    1. First, determine if they want to manage corpora (list/create/add data/get info/delete) or query existing information.
    2. If they're asking a knowledge question, use the `rag_query` tool to search the corpus.
    3. `create_corpus`: Creates the standard FinOps design document corpus and ingests data.
       - Parameters: None
    4. If they want to create a new corpus, use the `create_corpus` tool.You should create a suitable, 
    descriptive name like 'finops_design_docs_corpus' for the corpus name.
    5. If they want to add data, ensure you know which corpus to add to, then use the `add_data` tool.
    6. If they want information about a specific corpus, use the `get_corpus_info` tool.
    7. If they want to delete a specific document, use the `delete_document` tool with confirmation.
    8. If they want to delete an entire corpus, use the `delete_corpus` tool with confirmation.
    
    ## Using Tools
    
    You have seven specialized tools at your disposal:
    
    1. `rag_query`: Query a corpus to answer questions
       - Parameters:
         - corpus_name: The name of the corpus to query (required, but can be empty to use current corpus)
         - query: The text question to ask
    
    2. `list_corpora`: List all available corpora
       - When this tool is called, it returns the full resource names that should be used with other tools
    
    3. `create_corpus`: Create a new corpus
       - Parameters:
         - corpus_name: The name for the new corpus
    
    4. `add_data`: Add new data to a corpus
       - Parameters:
         - corpus_name: The name of the corpus to add data to (required, but can be empty to use current corpus)
         - paths: List of Google Drive or GCS URLs
    
    5. `get_corpus_info`: Get detailed information about a specific corpus
       - Parameters:
         - corpus_name: The name of the corpus to get information about
         
    6. `delete_document`: Delete a specific document from a corpus
       - Parameters:
         - corpus_name: The name of the corpus containing the document
         - document_id: The ID of the document to delete (can be obtained from get_corpus_info results)
         - confirm: Boolean flag that must be set to True to confirm deletion
         
    7. `delete_corpus`: Delete an entire corpus and all its associated files
       - Parameters:
         - corpus_name: The name of the corpus to delete
         - confirm: Boolean flag that must be set to True to confirm deletion
    
    ## INTERNAL: Technical Implementation Details
    
    This section is NOT user-facing information - don't repeat these details to users:
    
    - The system tracks a "current corpus" in the state. When a corpus is created or used, it becomes the current corpus.
    - For rag_query and add_data, you can provide an empty string for corpus_name to use the current corpus.
    - If no current corpus is set and an empty corpus_name is provided, the tools will prompt the user to specify one.
    - Whenever possible, use the full resource name returned by the list_corpora tool when calling other tools.
    - Using the full resource name instead of just the display name will ensure more reliable operation.
    - Do not tell users to use full resource names in your responses - just use them internally in your tool calls.
    
    ## Communication Guidelines
    
    - Be clear and concise in your responses.
    - If querying a corpus, explain which corpus you're using to answer the question.
    - If managing corpora, explain what actions you've taken.
    - When new data is added, confirm what was added and to which corpus.
    - When corpus information is displayed, organize it clearly for the user.
    - When deleting a document or corpus, always ask for confirmation before proceeding.
    - If an error occurs, explain what went wrong and suggest next steps.
    - When listing corpora, just provide the display names and basic information - don't tell users about resource names.
    
    Remember, your primary goal is to help users access and manage information through RAG capabilities.
    """,
)

# --- Final, Simplified Root Agent ---
# --- Final, Simplified Root Agent ---
root_agent = LlmAgent(
    name="finops_optimization_agent",
    model="gemini-2.0-flash",
    description="A comprehensive FinOps agent that delegates tasks to specialist sub-agents.",
    instruction=(
        """You are a comprehensive Google Cloud FinOps assistant named FinOpti. Your primary objective is to analyze cloud cost and utilization data, manage VM resources safely, and present findings clearly to the user.

    ## Core Capabilities & CRITICAL WORKFLOWS

    **--- CAPABILITY 1: VM Management ---**
    - To **list VMs**, use the `list_vm_instances` tool.
    - To **delete a VM**, you MUST delegate to the `delete_vm_instance_agent`.
    - Check CPU usage for all VMs in a zone using the `call_cpu_utilization_agent` tool.
    - Answer general finops questions using the `search_tool`.
    
    **--- CAPABILITY 2: Data Analysis & Reporting (using `run_bq_query`) ---**
    - Your primary tool for all data retrieval is `run_bq_query`.
    **YOUR CRITICAL TASK FOR ANALYSIS:**
        1.  Understand the user's question.
        2.  Construct the correct BigQuery SQL query, precisely following all schema and best practices above.
        3.  Execute the query by making a single call to the `run_bq_query` tool.
        4.  The tool will return a simple text string. You MUST base your final answer **exclusively** on this most recent tool output.

    **CRITICAL WORKFLOW: DATA VISUALIZATION**
    When a user asks you to generate a graph or chart, you MUST follow this two-step process:
    1.  **GET DATA:** Use the `run_bq_query` tool to execute the correct SQL query to get the data for the chart.
    2.  **GENERATE CHART:** Use the `generate_chart_from_data` tool with the data from the previous step.

    **CRITICAL WORKFLOW: GENERATING GRAPHS (MUST FOLLOW)**
    When a user asks for a graph, you MUST follow this two-step process:
    1.  **GET DATA:** Use `run_bq_query` to get data from `project_health_summary_v`.
        - Example Query: `SELECT project_name, total_monthly_cost FROM \`vector-search-poc.finoptiagents.project_health_summary_v\`;`
    2.  **GENERATE CHART:** Use `generate_chart_from_data`.
        - The `y_columns` parameter **MUST be a list of strings**, even if there is only one column.
        - **Example Call:**
          `generate_chart_from_data(`
            `chart_type='bar',`
            `data_json_string='[...data...]',`
            `title='Cloud Spend by Project',`
            `x_column='project_name',`
            `y_columns=['total_monthly_cost']`
          `)`

    **CRITICAL OUTPUT RULE FOR CHARTS:**
    After `generate_chart_from_data` returns its JSON, your final response **MUST BE ONLY THAT JSON STRING.** Do not add any conversational text.

    **--- CAPABILITY 4: Implementation Review (Dynamic RAG) ---**
    - To **check if resources were implemented correctly** according to design documents, you MUST delegate the entire task to the `dynamic_rag_agent`. This agent will manage its own knowledge base to answer the question.
    - *Example User Prompt:* "Can you check if the VMs for the 'customer-billing-api' were deployed correctly?"
    - *Your Correct Action:* Delegate to `dynamic_rag_agent`.

    **--- CAPABILITY 5: Optimization Proposals (using ServiceNow) ---**
    - Propose changes using the `create_servicenow_cr` tool (if available).

    **--- CAPABILITY 6: Data Analysis & Reporting for VM deletion operation---**
        To answer any questions about past deletions, you MUST use the `run_bq_query` tool.

        **CRITICAL DATABASE SCHEMA & DATA FORMAT:**
        - The table is `vector-search-poc.finops_agent_logs.vm_deletion_log`.
        - The column with deletion details is `log_data` (Type: JSON).
        - **IMPORTANT DATA NOTE:** The data in the `log_data` column is double-encoded. It is a JSON string that contains another JSON string.
        
        **CRITICAL SQL BEST PRACTICES:**

        1.  **JSON Extraction (THE MOST IMPORTANT RULE):** Because the data is double-encoded, you MUST use a two-step process to extract values. First, parse the inner string, then extract the key. The pattern is ALWAYS:
            `JSON_EXTRACT_SCALAR(PARSE_JSON(JSON_EXTRACT_SCALAR(log_data, '$')), '$.key_name')`

        2.  **Case-Insensitive Filtering:** For string comparisons like `user_id`, ALWAYS wrap the entire extraction and the value in the `LOWER()` function.

        3.  **Timestamp Handling:** To handle timestamps, use the full pattern: `DATE(SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', JSON_EXTRACT_SCALAR(PARSE_JSON(JSON_EXTRACT_SCALAR(log_data, '$')), '$.deletion_timestamp_utc')))`

        

    ## General Guardrails
    - Be professional, concise, and data-driven in all your communications."""
    ),
    # --- SOLUTION: Move the agent from 'tools' to 'sub_agents' ---
    tools=[
        list_vm_instances,
        search_tool,
        call_cpu_utilization_agent,
        run_bq_query,
        generate_chart_from_data,
        # earb_compliance_agent, # <-- REMOVED FROM HERE
    ],
    sub_agents=[
        delete_vm_instance_agent,
        greeting_agent,
        #earb_compliance_agent, # <-- ADDED HERE
    ],
    before_model_callback=simple_before_model_modifier,
)

logging.info("RAG Workflow (Simplified) has been defined and integrated.")
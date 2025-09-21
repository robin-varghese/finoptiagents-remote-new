from google.adk.agents import Agent,LoopAgent,BaseAgent,LlmAgent, SequentialAgent
from google.adk.sessions import DatabaseSessionService, Session
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.sessions import VertexAiSessionService
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.bigquery import BigQueryToolset, BigQueryCredentialsConfig

from pydantic import BaseModel # Or from wherever ADK makes it accessible
from typing import Optional, List # ### --- MODIFIED --- ### Add List
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
import uuid # For generating unique IDs or for user_id if not available elsewhere
from toolbox_core import ToolboxClient
from google.adk.tools import ToolContext
from google.cloud import bigquery
from .utils.embeddings import generate_combined_embedding
import asyncio 

#from toolbox_langchain import ToolboxClient
#START------>project configurations<------------
from google.cloud import secretmanager
from google.api_core import exceptions
import google.auth

def _get_secret_value(project_id: str, secret_id: str, client: secretmanager.SecretManagerServiceClient) -> str | None:
    """Fetches a secret's value, returning None if not found or on error."""
    if not project_id:
        print(f"Warning: Project ID is not set. Cannot fetch secret '{secret_id}'.")
        return None

    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    try:
        response = client.access_secret_version(request={"name": name})
        value = response.payload.data.decode("UTF-8")
        print(f"Successfully fetched secret: '{secret_id}'")
        return value
    except exceptions.NotFound:
        print(f"Warning: Secret '{secret_id}' not found in project '{project_id}'.")
        return None
    except Exception as e:
        print(f"Warning: An error occurred while fetching secret '{secret_id}': {e}")
        return None

print("--- Loading configuration from Google Secret Manager ---")

# 1. Initialize client and determine the project ID to use for fetching secrets.
_secret_client = secretmanager.SecretManagerServiceClient()
_initial_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not _initial_project_id:
    try:
        _, _initial_project_id = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        _initial_project_id = None

# 2. The definitive project ID can be stored in a secret named 'google-project-id'.
# We use the initial project ID to try and fetch it.
# If it's not found, we fall back to the initial project ID.
GOOGLE_PROJECT_ID = _get_secret_value(_initial_project_id, "google-project-id", _secret_client) or _initial_project_id

if not GOOGLE_PROJECT_ID:
    # If we still don't have a project ID, we can't proceed.
    raise ValueError(
        "FATAL: Could not determine Google Cloud Project ID. "
        "Set the GOOGLE_CLOUD_PROJECT environment variable or run 'gcloud auth application-default login', "
        "or ensure a secret named 'google-project-id' exists."
    )

print(f"Using Project ID for secrets: {GOOGLE_PROJECT_ID}")

# 3. Define a helper to fetch other secrets using the definitive project ID.
def _fetch_config(secret_id: str, default: str | None = None) -> str | None:
    """Fetches a secret using the determined GOOGLE_PROJECT_ID."""
    value = _get_secret_value(GOOGLE_PROJECT_ID, secret_id, _secret_client)
    return value if value is not None else default

# 4. Fetch all configuration values. Secret IDs are lower-case and hyphenated.
GOOGLE_GENAI_USE_VERTEXAI = _fetch_config("google-genai-use-vertexai")
GOOGLE_API_KEY = _fetch_config("google-api-key")
GOOGLE_ZONE = _fetch_config("google-zone")
STAGING_BUCKET_URI = _fetch_config("staging-bucket-uri")
PROD_BUCKET_URI = _fetch_config("prod-bucket-uri")
PACKAGE_URI = _fetch_config("package-uri")
GOOGLE_DB_URI = _fetch_config("google-db-uri")
OPENAI_API_KEY = _fetch_config("openai-api-key")
#MCP_TOOLBOX_URL = _fetch_config("mcp-toolbox-url")
TOOLSET_NAME_FOR_LOGGING = _fetch_config("toolset-name-for-logging")
LOGGING_TOOL_NAME = _fetch_config("logging-tool-name")
BIGQUERY_DATASET_ID = _fetch_config("bigquery-dataset-id")
BIGQUERY_TABLE_ID = _fetch_config("bigquery-table-id")
REMOTE_CPU_AGENT_RESOURCE_NAME = _fetch_config("remote-cpu-agent-resource-name")

# Clean up temporary variables from the global scope
del _get_secret_value, _fetch_config, _secret_client, _initial_project_id

print("--- Configuration loading complete. ---")
#END------>project configurations<------------

load_dotenv()

# Initialize Vertex AI SDK - This is crucial for ReasoningEngine to work
if GOOGLE_PROJECT_ID and GOOGLE_ZONE:
    # Reasoning Engines are regional, so we extract the region from the zone
    # e.g., 'us-central1-a' -> 'us-central1'
    google_region = "-".join(GOOGLE_ZONE.split("-")[:-1])
    vertexai.init(
        project=GOOGLE_PROJECT_ID, location=google_region
    )
    print(f"Vertex AI initialized for project '{GOOGLE_PROJECT_ID}' in region '{google_region}'")
else:
    print("Skipping Vertex AI initialization. GOOGLE_PROJECT_ID and/or GOOGLE_ZONE not set in .env file.")

# --- 1. Define Constants ---
APP_NAME = "agent_comparison_app"
USER_ID = "Robin Varghese"
BASE_SESSION_ID_TOOL_AGENT = "session_tool_agent_xyz"
SESSION_ID_SCHEMA_AGENT = "session_schema_agent_xyz"
current_session_id_tool_agent = BASE_SESSION_ID_TOOL_AGENT + str(time.time())
MODEL_NAME = "gemini-2.0-flash"

# Define a minimal Pydantic model for event content if no specific fields are needed
class EmptyEventContent(BaseModel):
    pass

client = None  # Initialize as None to be created lazily inside the async callback.
my_tools = None  # Will hold the loaded toolset object.
useraction_insert_mcptool = None  # Will hold the name of the logging tool.

#*************************START: TOOLS Section**************************************

# In agent.py, REPLACE the existing delete_vm_instance tool with this one.

# In agent.py, REPLACE your existing delete_vm_instance tool with this one.

def delete_vm_instance(project_id: str, instance_id: str, zone: str, tool_context: ToolContext):
    """
    Deletes a VM instance and AUTOMATICALLY logs the deletion event to BigQuery upon success.
    This is now an atomic operation.
    """
    print(f"--- [Tool] Attempting to delete VM: '{instance_id}' in project '{project_id}' ---")
    headers = {'Content-Type': 'application/json'}
    data = {'instance_id': instance_id, 'project_id': project_id, 'zone': zone}
    url = f"https://agent-tools-912533822336.us-central1.run.app/delete_vms"

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_data = response.json()

        # Check for the specific success status from your API response
        is_deleted = (
            response_data.get("results") and
            response_data["results"][0].get("status") == "deleted"
        )

        if is_deleted:
            print(f"API confirmed successful deletion of '{instance_id}'.")

            # --- Triggering automatic BigQuery logging... ---
            print("--- Triggering automatic BigQuery logging... ---")
            
            # --- THIS IS THE CORRECTED LINE ---
            # First, try to get user from session state. If it's not there,
            # fall back to the globally defined USER_ID constant.
            current_user_id = tool_context.state.get("user_name") or USER_ID
            # --- END CORRECTION ---

            log_status = log_vm_deletion_to_bigquery(
                project_id=project_id,
                instance_id=instance_id,
                zone=zone,
                user_id=current_user_id, # Pass the corrected user ID
                tool_context=tool_context
            )
            print(f"--- BigQuery logging status: {log_status} ---")

            return response_data
        else:
            print(f"API reported failure to delete '{instance_id}': {response_data}")
            return response_data

    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling deletion API: {e}"
        print(error_msg)
        return {"status": "error", "message": str(e)}

def list_vm_instances(project_id: str, zone: str):
    """Lists VM instances based on domain, project ID, and zone using the /list_vms endpoint.

    Args:
        project_id: The Google Cloud project ID.
        zone: The zone where the instances are located.
        service_url: The URL of the Cloud Run service.

    Returns:
        The JSON response from the API, or None if an error occurs.
    """
    print(f" I am inside list_vm_instances 'project_id': {project_id}, 'zone': {zone}")
    headers = {'Content-Type': 'application/json'}
    data = {'project_id': project_id, 'zone': zone}
    url = f"https://agent-tools-912533822336.us-central1.run.app/list_vms"

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error listing instances: {e}")
        return None

def search_tool(query: str):
    CLOUD_RUN_URL = "https://ddsearchlangcagent-qcdyf5u6mq-uc.a.run.app/search"
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    print(f"Sending POST request to: {CLOUD_RUN_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}") 

    try:
        response = requests.post(CLOUD_RUN_URL, headers=headers, json=payload, timeout=120) 
        response.raise_for_status() 
        result_data = response.json()
        print("\n--- Agent Response ---")
        print(result_data)
        return result_data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON response from the server.")
        return {"error": "Invalid JSON response from server."}


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
        return f"Error during synchronous stream call: {str(e)}"

async def call_cpu_utilization_agent(project_id: str, zone: str) -> str:
    """
    Asynchronously calls the remote Agent Engine agent by running the
    synchronous stream iteration in a separate thread.
    """
    print(f"--> [Local Agent Tool] Calling remote agent via asyncio.to_thread")
    if not REMOTE_CPU_AGENT_RESOURCE_NAME:
        return "Error: REMOTE_CPU_AGENT_RESOURCE_NAME is not set in the environment."
    try:
        query = f"What is the CPU utilization for all VMs in project {project_id} and zone {zone}?"
        final_response = await asyncio.to_thread(
            _get_streamed_response_sync,
            query,
            REMOTE_CPU_AGENT_RESOURCE_NAME
        )
        print(f"<-- [Remote Agent Final Response] {final_response}")
        return final_response
    except Exception as e:
        print(f"Error in async tool 'call_cpu_utilization_agent': {e}")
        return f"An unexpected error occurred in the async tool wrapper: {str(e)}"

### --- NEW --- ###
# In agent.py, inside the #*************************START: TOOLS Section**************************************

### --- NEW TOOL --- ###
# This tool prepares the log data, simplifying the logger agent's job.
# In agent.py, add this new function inside the TOOLS section

def log_vm_deletion_to_bigquery(
    project_id: str,
    instance_id: str,
    zone: str,
    user_id: str,
    tool_context: ToolContext
) -> str:
    """
    Constructs a log entry for a VM deletion, generates an embedding,
    and inserts it into the BigQuery log table. This is a single,
    comprehensive tool for logging.
    """
    print(f"--- [Tool] log_vm_deletion_to_bigquery called for VM: {instance_id} ---")
    
    # 1. Create the JSON log data
    log_payload = {
        "user_id": user_id,
        "deletion_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "vm_id": instance_id,
        "project_id": project_id,
        "zone": zone,
    }
    log_json_string = json.dumps(log_payload)
    print(f"Generated Log JSON: {log_json_string}")

    # 2. Generate the embedding for the JSON string
    embedding_vector = generate_combined_embedding(text_to_embed=log_json_string)
    if not embedding_vector:
        error_msg = "Failed to generate embedding. Aborting log insertion."
        print(error_msg)
        return error_msg

    # 3. Construct and execute the SQL insert statement
    # The table ID must be enclosed in backticks ``
    table_id = "`vector-search-poc.finops_agent_logs.vm_deletion_log`"
    
    # We use query parameters for safety and correctness
    sql_query = f"""
        INSERT INTO {table_id} (log_data, embedding)
        VALUES (@log_data_json, @embedding_vector)
    """
    
    try:
        # It's best practice to initialize the client inside the tool
        # to ensure thread safety and proper credentials.
        bq_client = bigquery.Client(project=GOOGLE_PROJECT_ID)
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("log_data_json", "JSON", log_json_string),
                bigquery.ArrayQueryParameter("embedding_vector", "FLOAT64", embedding_vector),
            ]
        )
        
        print(f"Executing BigQuery INSERT statement...")
        query_job = bq_client.query(sql_query, job_config=job_config)
        query_job.result()  # Wait for the job to complete

        success_msg = f"Successfully logged deletion of VM {instance_id} to BigQuery."
        print(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"An error occurred during BigQuery insert: {e}"
        print(error_msg)
        return error_msg
# New tool to generate embeddings for logging. This acts as a bridge
# between the agent's reasoning and the embedding utility function.

# In agent.py, add this new tool to your TOOLS Section

# In agent.py, REPLACE your existing run_bq_query_and_get_simple_answer tool

# In agent.py, in the TOOLS Section, ADD this new function.
# This replaces the BigQueryToolset and the wrapper.

# In agent.py, REPLACE the entire run_bq_query tool with this definitive version.

def run_bq_query(query: str, project_id: str) -> str:
    """
    Executes a read-only BigQuery query and returns the result as a simple,
    human-readable string. Use this for all data analysis and reporting.
    """
    print(f"--- [Standalone BQ Tool] Executing query for project '{project_id}': {query} ---")
    try:
        client = bigquery.Client(project=project_id)
        query_job = client.query(query)
        results = query_job.result()

        if results.total_rows == 0:
            return "The query returned no results."

        # --- THIS IS THE CRITICAL FIX ---
        # Robustly handle single-value results (like COUNT)
        if results.total_rows == 1:
            # Get the first row from the iterator
            first_row = next(iter(results))
            # Get the first value from that row, regardless of column name
            first_value = first_row[0]
            
            # Explicitly cast to int, then to string to ensure a clean result
            try:
                result_as_int = int(first_value)
                return f"The result is: {str(result_as_int)}"
            except (ValueError, TypeError):
                # If it's not a number, just return it as a string
                return f"The result is: {str(first_value)}"
        # --- END CRITICAL FIX ---

        # Handle queries returning multiple rows/columns
        output_rows = []
        # Re-run the query to reset the iterator after the check above
        results = client.query(query).result()
        for row in results:
            output_rows.append(str(dict(row)))
            if len(output_rows) >= 5:
                break
        
        return f"Query returned {results.total_rows} rows. Here are the first few:\n" + "\n".join(output_rows)

    except Exception as e:
        print(f"--- [Standalone BQ Tool] ERROR: {e} ---")
        return f"An error occurred while running the query: {e}"
#*************************END: TOOLS Section**************************************


#*************************START: Call Back ***************************************
def simple_before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Inspects/modifies the LLM request or skips the call.
    This version safely handles system_instruction as a str or Content object.
    """
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

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
    print(f"[Callback] Modified system instruction to: '{modified_text}'")

    last_user_message = ""
    if llm_request.contents:
        last_content_item = llm_request.contents[-1]
        if last_content_item.role == 'user' and last_content_item.parts:
            if last_content_item.parts[0].text is not None:
                last_user_message = last_content_item.parts[0].text

    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    if "BLOCK" in last_user_message.upper():
        print("[Callback] 'BLOCK' keyword found. Skipping LLM call.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    else:
        print("[Callback] Proceeding with LLM call.")
        return None
    
#*************************END: Call Back *****************************************

# Initialize the BigQuery toolset
application_default_credentials, _ = google.auth.default()
credentials_config = BigQueryCredentialsConfig(
    credentials=application_default_credentials
)
# In agent.py, find and update the BigQueryToolset initialization

#*************************START: Agents Section**************************************

# In agent.py, replace the old delete_vm_instance_agent with this new one.
# All other agents, including the root_agent, remain the same.

# In agent.py, replace the delete_vm_instance_agent with this new, smarter version.

delete_vm_instance_agent = LlmAgent(
    name="delete_vm_instance_agent",
    model="gemini-2.0-flash",
    description="A careful agent that verifies a VM exists and then calls a single tool to delete and log the action.",
    instruction="""You are a careful, two-step agent for deleting a VM.
    1.  **VERIFY:** Read the user's request to get the `project_id`, `zone`, and `instance_id`. Your first action MUST be to call `list_vm_instances` to confirm the VM exists.
    2.  **EXECUTE:** If the VM is in the list, your second action MUST be to call the `delete_vm_instance` tool. This single tool will handle both the deletion and the logging automatically. If the VM is not in the list, inform the user they may have provided incorrect details.
    """,
    tools=[list_vm_instances, delete_vm_instance],
)

### --- NEW --- ###
# This new agent is responsible for logging the deletion to BigQuery.
# It runs as the second step in the deletion_workflow_agent sequence.
# In agent.py, replace the old vm_deletion_logger_agent with this one.

# In agent.py, replace the old logger agent with this simplified one.

greeting_agent = LlmAgent(
    name="Greeter",
    description=
    """This agent should greet the user when logged-in""",
    model="gemini-2.0-flash", # Use a valid model
    instruction="Generate a short, friendly greeting.",
    output_key="last_greeting"
)

### --- MODIFIED --- ###
# The root agent is updated to use the new `deletion_workflow_agent`.
# In agent.py, replace the root_agent with this one.

root_agent = LlmAgent(
    name="finops_optimization_agent",
    model="gemini-2.0-flash",
    description=(
        "A comprehensive FinOps agent that can delete VMs and perform data analysis on deletion logs using its built-in BigQuery tool."
    ),
    instruction=(
        """You are a comprehensive Google Cloud FinOps assistant. You have two primary capabilities: managing VMs and analyzing deletion logs.

        **--- CAPABILITY 1: VM Management ---**
        - To **list VMs**, use the `list_vm_instances` tool.
        - To **delete a VM**, you MUST delegate to the `delete_vm_instance_agent`.

        **--- CAPABILITY 2: Data Analysis & Reporting ---**
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

        **YOUR CRITICAL TASK FOR ANALYSIS:**
        1.  Understand the user's question.
        2.  Construct the correct BigQuery SQL query, precisely following all schema and best practices above.
        3.  Execute the query by making a single call to the `run_bq_query` tool.
        4.  The tool will return a simple text string. You MUST base your final answer **exclusively** on this most recent tool output.

        **--- PERFECT QUERY EXAMPLE ---**
        - User: "how many vms were deleted by Robin Varghese"
        - Your Action: `run_bq_query(query="SELECT count(*) FROM `vector-search-poc.finops_agent_logs.vm_deletion_log` WHERE LOWER(JSON_EXTRACT_SCALAR(PARSE_JSON(JSON_EXTRACT_SCALAR(log_data, '$')), '$.user_id')) = LOWER('Robin Varghese')", project_id="vector-search-poc")`
        """
    ),
    tools=[
        list_vm_instances,
        search_tool,
        call_cpu_utilization_agent,
        run_bq_query
    ],
    sub_agents=[
        delete_vm_instance_agent,
        greeting_agent,
    ],
    before_model_callback=simple_before_model_modifier,
)
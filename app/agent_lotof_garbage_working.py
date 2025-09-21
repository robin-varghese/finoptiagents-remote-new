from google.adk.agents import Agent,LoopAgent,BaseAgent,LlmAgent, SequentialAgent
from google.adk.sessions import DatabaseSessionService, Session
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.sessions import VertexAiSessionService
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.bigquery import BigQueryToolset, BigQueryCredentialsConfig

from pydantic import BaseModel # Or from wherever ADK makes it accessible
from typing import Optional
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
import asyncio # <-- Add this import

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

def delete_vm_instance(project_id: str, instance_id: str, zone: str, tool_context: ToolContext):
    """Deletes a VM instance using the /delete_vms endpoint and saves its info to state.

    Args:
        project_id: The Google Cloud project ID.
        instance_id: The ID of the instance to delete.
        zone: The zone where the instance is located.

    Returns:
        The JSON response from the API, or None if an error occurs.
    """
    print(f" I am inside delete_vm_instances")
    headers = {'Content-Type': 'application/json'}
    data = {'instance_id': instance_id, 'project_id': project_id, 'zone': zone}
    print(f"'instance_id': {instance_id}, 'project_id': {project_id}, 'zone': {zone}")
    url = f"https://agent-tools-912533822336.us-central1.run.app/delete_vms"
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        print(f"VM Deletion successful: {response_data}")
        # Save details for the logging agent
        tool_context.state["last_deleted_vm"] = {
            "instance_id": instance_id,
            "project_id": project_id,
            "zone": zone,
        }
        return response_data
    except requests.exceptions.RequestException as e:
        print(f"Error deleting instance: {e}")
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

# Create a DuckDuckGo search tool
def search_tool(query: str):
    # --- Configuration ---
    # The URL of your deployed Cloud Run service endpoint
    # Ensure it includes the specific path (/search)
    CLOUD_RUN_URL = "https://ddsearchlangcagent-qcdyf5u6mq-uc.a.run.app/search"

    # The query you want to send to the agent
    #search_query = "What are the latest developments in AI regulation in Europe?"

    # Optional: If your agent uses chat history, prepare it
    # This should match the structure expected by your format_chat_history helper
    chat_history_example = [
        {"role": "user", "content": "Tell me about large language models."},
        {"role": "assistant", "content": "Large language models are advanced AI systems..."}
    ]

    # --- Prepare the Request ---
    # This structure MUST match the Pydantic model `SearchRequest` in your FastAPI app
    payload = {
        "query": query,
        # Uncomment and include if your agent uses history:
        # "chat_history": chat_history_example
    }

    # Set the headers for sending JSON data
    headers = {
        "Content-Type": "application/json"
    }

    # --- Make the API Call ---
    print(f"Sending POST request to: {CLOUD_RUN_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}") # Log the payload being sent

    try:
        # Send the POST request
        response = requests.post(CLOUD_RUN_URL, headers=headers, json=payload, timeout=120) # Set a reasonable timeout (in seconds)

        # --- Handle the Response ---
        # Check if the request was successful (status code 2xx)
        response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)

        # Parse the JSON response from the server
        result_data = response.json() # This should match the `SearchResponse` model

        # Extract the result
        #agent_response = result_data.get("result", "No 'result' field found in response.")

        print("\n--- Agent Response ---")
        print(result_data)
        return result_data

    except requests.exceptions.Timeout:
        print(f"Error: The request to {CLOUD_RUN_URL} timed out.")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Status Code: {response.status_code}")
        # Try to print the error detail from the server response if available
        try:
            error_detail = response.json()
            print(f"Server Error Detail: {error_detail}")
        except json.JSONDecodeError:
            print(f"Server Response (non-JSON): {response.text}")
    except requests.exceptions.RequestException as req_err:
        # Catch other potential errors like connection errors, etc.
        print(f"An error occurred during the request: {req_err}")
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON response from the server.")
        print(f"Response Text: {response.text}") # Print raw text if JSON decoding fails
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Add this new helper function
# In your local orchestrator's agent.py file
# In your local orchestrator's agent.py file
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
        
        # Use a regular 'for' loop
        for event in stream:
            print(f"Received stream event: {event}")

            # --- FINAL FIX: Correctly parse the 'content' dictionary ---
            # Check if the 'content' key exists and is a dictionary
            if isinstance(event.get("content"), dict):
                content = event["content"]
                # Check if 'parts' exists and is a list
                if isinstance(content.get("parts"), list):
                    for part in content["parts"]:
                        # Check if the part is a dictionary and has a 'text' key
                        if isinstance(part, dict) and "text" in part:
                            text_chunk = part["text"]
                            if text_chunk:
                                print(f"Extracted text chunk: {text_chunk}")
                                response_parts.append(text_chunk)
        
        final_response = "".join(response_parts).strip()
        
        # Check if we actually got a response before returning
        if not final_response:
             print("WARNING: No text parts found in any event from the stream.")
             return "No text response could be parsed from the remote agent's stream."
        
        return final_response

    except Exception as e:
        print(f"Error inside synchronous stream helper: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during synchronous stream call: {str(e)}"
# This is the new, correct implementation that mimics your working notebook
# This tool remains an 'async def'
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
        
        # Run the synchronous helper function in a separate thread
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

# { ... your existing tools like delete_vm_instance, list_vm_instances, etc. ... }
#*************************END: TOOLS Section**************************************

# --- New Deletion Logging Components ---

# async def log_last_vm_deletion(tool_context: ToolContext) -> dict:
#     """
#     Reads the details of the most recently deleted VM from the session state
#     and logs this information directly to Google BigQuery.
#     """
#     print("[Tool] log_last_vm_deletion triggered for BigQuery.")
#     last_deleted_vm = tool_context.state.get("last_deleted_vm")

#     if not last_deleted_vm:
#         print("[Tool] No 'last_deleted_vm' key found in state. Nothing to log.")
#         return {"status": "skipped", "message": "No VM deletion information found in state."}

#     try:
#         project_id = last_deleted_vm.get('project_id')
#         if not project_id:
#              return {"status": "error", "message": "Project ID not found for logging."}

#         # Initialize BigQuery client. It will use application default credentials.
#         bq_client = bigquery.Client(project=project_id)

#         # Prepare the data in the requested pipe-separated format
#         now = datetime.datetime.now(ZoneInfo("UTC"))
#         log_entry_string = (
#             f"{tool_context.session.id}|"
#             f"{tool_context.session.user_id or USER_ID}|"
#             f"{now.strftime('%Y-%m-%d')}|"
#             f"{now.strftime('%H:%M:%S')}|"
#             f"{last_deleted_vm.get('instance_id')}|"
#             f"|"  # Two pipes as requested
#             f"{last_deleted_vm.get('zone')}"
#         )

#         # Prepare the row for insertion.
#         # Assumes the table has a single column named 'log_entry' of type STRING.
#         rows_to_insert = [{"log_entry": log_entry_string}]
#         table_ref = bq_client.dataset(BIGQUERY_DATASET_ID).table(BIGQUERY_TABLE_ID)

#         print(f"[Tool] Inserting log into BigQuery: {project_id}.{BIGQUERY_DATASET_ID}.{BIGQUERY_TABLE_ID}")
#         print(f"[Tool] Log payload: {rows_to_insert}")

#         errors = bq_client.insert_rows_json(table_ref, rows_to_insert)

#         if not errors:
#             print("[Tool] Log successfully inserted into BigQuery.")
#             tool_context.state.pop("last_deleted_vm", None) # Clear state to avoid re-logging
#             return {"status": "success", "message": "Log entry saved to BigQuery."}
#         else:
#             print(f"[Tool] Encountered errors while inserting rows into BigQuery: {errors}")
#             return {"status": "error", "message": f"BigQuery insertion errors: {errors}"}

#     except Exception as e:
#         print(f"Error in log_last_vm_deletion tool: {e}")
#         import traceback
#         traceback.print_exc()
#         return {"status": "error", "message": str(e)}
#*************************START: Call BAck ***************************************
def simple_before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Inspects/modifies the LLM request or skips the call.
    This version safely handles system_instruction as a str or Content object.
    """
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

    # --- Modification Example ---
    # Add a prefix to the system instruction safely, handling both str and Content types.
    prefix = "[Modified by Callback] "
    current_instruction = llm_request.config.system_instruction

    base_text = ""
    if isinstance(current_instruction, str):
        base_text = current_instruction
    elif isinstance(current_instruction, types.Content) and current_instruction.parts:
        # Use the text from the first part if it exists
        base_text = current_instruction.parts[0].text or ""
    # If current_instruction is None or an empty Content object, base_text remains ""

    modified_text = prefix + base_text

    # Create a new Content object and assign it back. This is the safest way.
    llm_request.config.system_instruction = types.Content(
        role="system",
        parts=[types.Part(text=modified_text)]
    )
    print(f"[Callback] Modified system instruction to: '{modified_text}'")

    # --- Inspect last user message ---
    last_user_message = ""
    if llm_request.contents:
        last_content_item = llm_request.contents[-1]
        if last_content_item.role == 'user' and last_content_item.parts:
            if last_content_item.parts[0].text is not None:
                last_user_message = last_content_item.parts[0].text

    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    # --- Skip Example ---
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
    
#*************************START: Call Back ***************************************
# (Global initializations for client/my_tools, USER_ID, LOGGING_TOOL_NAME, generate_combined_embedding remain)
# Ensure 'client' is your initialized ToolboxClient instance, not 'my_tools' for calling methods.
# I've changed 'my_tools' back to 'client' in the logging check and call_tool.

# Ensure 'toolbox_client' is defined and initialized at the module level as previously discussed
# For example:
# try:
#     toolbox_client = ToolboxClient(os.getenv("MCP_TOOLBOX_URL"))
#     toolbox_client.load_toolset(os.getenv("TOOLSET_NAME_FOR_LOGGING", "my_googleaiagent_toolset"))
#     print("ToolboxClient initialized and toolset loaded.")
# except Exception as e:
#     print(f"CRITICAL: Failed to initialize ToolboxClient: {e}")
#     toolbox_client = None


# Find this function in your code and replace it completely with the following:

# --- MODIFIED: Update the logging callback ---
# async def log_interaction_after_model(
#     callback_context: CallbackContext,
#     llm_response: LlmResponse
# ) -> None:
#     """
#     Asynchronously logs the LLM response. Lazily initializes the ToolboxClient
#     and loads the toolset on the first run.
#     """
#     global client, my_tools, useraction_insert_mcptool
#     print("[Callback] After model call triggered.")

#     # --- LAZY-LOADING AND INITIALIZATION LOGIC ---
#     # Initialize the ToolboxClient on the first call, inside the running event loop.
#     if client is None:
#         print("[Callback] First run: Initializing ToolboxClient...")
    #     try:
    #         mcp_url = MCP_TOOLBOX_URL
    #         if not mcp_url:
    #             print("[Callback] MCP_TOOLBOX_URL not set. Disabling logging.")
    #             my_tools = "FAILED_TO_LOAD"  # Mark as failed to prevent retries
    #             return
    #         client = ToolboxClient(mcp_url)
    #     except Exception as e:
    #         print(f"[Callback] CRITICAL ERROR initializing ToolboxClient: {e}")
    #         my_tools = "FAILED_TO_LOAD"
    #         return

    # # Lazy-load the toolset on the first call
    # if my_tools is None:
    #     print("[Callback] First run: Loading MCP toolset asynchronously...")
    #     try:
    #         loaded_tools = await client.load_toolset(TOOLSET_NAME_FOR_LOGGING)
    #         if loaded_tools:
    #             my_tools = loaded_tools
    #             useraction_insert_mcptool = LOGGING_TOOL_NAME
    #             print(f"[Callback] MCP toolset loaded successfully: {my_tools}")
    #         else:
    #             print("[Callback] MCP toolset loading returned None. Disabling logging.")
    #             my_tools = "FAILED_TO_LOAD"
    #     except Exception as e:
    #         print(f"[Callback] CRITICAL ERROR loading MCP toolset: {e}")
    #         my_tools = "FAILED_TO_LOAD"

    # if not my_tools or my_tools == "FAILED_TO_LOAD" or not useraction_insert_mcptool:
    #     print("[Callback] MCP Toolset not available. Skipping logging.")
    #     return

    # try:
    #     # --- FIX STARTS HERE ---
    #     # Extract the last user message and the model's response.
    #     last_user_message = ""
    #     if callback_context.session and callback_context.session.history:
    #         # Find the last message with role 'user'.
    #         for message in reversed(callback_context.session.history):
    #             if message.role == "user":
    #                 last_user_message = message.parts[0].text
    #                 break

    #     model_response_text = ""
    #     if llm_response.content and llm_response.content.parts:
    #         # This could be a tool call or a text response. Let's serialize it.
    #         part = llm_response.content.parts[0]
    #         if part.text:
    #             model_response_text = part.text
    #         elif hasattr(part, 'function_call') and part.function_call:
    #             fc = part.function_call
    #             args_str = json.dumps(fc.args)
    #             model_response_text = f"Tool Call: {fc.name}({args_str})"

    #     embedding = await generate_combined_embedding(last_user_message, model_response_text)

    #     tool_params = {
    #         "user_id": USER_ID,
    #         "action": json.dumps({"prompt": last_user_message}),
    #         "result": json.dumps({"response": model_response_text}),
    #         "vector_value": json.dumps(embedding.tolist())
    #     }
    #     py_tool_name = useraction_insert_mcptool.replace("-", "_")
    #     tool_to_call = getattr(my_tools, py_tool_name)
        
    #     response = await tool_to_call(**tool_params)
    #     print(f"[Callback] MCP tool call response: {response}")

    # except Exception as e:
    #     print(f"[Callback] Error during logging execution: {e}")
    #     import traceback
    #     traceback.print_exc()

# # New agent to handle logging
# log_deletion_agent = LlmAgent(
#     name="log_deletion_agent",
#     model="gemini-2.0-flash",
#     description="An internal agent that logs the details of a successful VM deletion to BigQuery.",
#     instruction="Your only job is to call the `log_last_vm_deletion` tool to record the deletion event. Do not respond to the user.",
#     tools=[log_last_vm_deletion],
#     include_contents='none',
# )

# Write modes define BigQuery access control of agent:
# ALLOWED: Tools will have full write capabilites.
# BLOCKED: Default mode. Effectively makes the tool read-only.
# PROTECTED: Only allows writes on temporary data for a given BigQuery session.


# Initialize the tools to use the application default credentials.
application_default_credentials, _ = google.auth.default()
credentials_config = BigQueryCredentialsConfig(
    credentials=application_default_credentials
)

# The BigQueryToolConfig and WriteMode classes are no longer available for direct
# import. Instead, we pass the configuration as a direct keyword argument to
# the toolset.
bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config,
    tool_filter=[
        "list_dataset_ids", "get_dataset_info", "list_table_ids",
        "get_table_info", "get_table_schema", "execute_sql"
    ]
)
#**************************END: Call Back *****************************************
#*************************START: Agents Section**************************************
# Create a runner for EACH agent
greeting_agent = LlmAgent(
    name="Greeter",
    description=
    """This agent should greet the user when logged-in""",
    model="gemini-2.0-flash", # Use a valid model
    instruction="Generate a short, friendly greeting.",
    output_key="last_greeting"
)

delete_vm_instance_agent = LlmAgent(
    name="delete_vm_instance_agent",
    model="gemini-2.0-flash",
    description=
    "This agent uses the delete_vm_instance tool to delete a single VM. It is for internal use by other agents.",
    instruction="""You are a single-purpose agent. Your only job is to call the `delete_vm_instance` tool.
The details of the VM to delete are in the state variable 'current_vm_details'.
Extract the 'project_id', 'instance_id', and 'zone' from the 'current_vm_details' state variable and use them to call the tool.
Do not add any commentary or ask for confirmation.""",
    tools=[delete_vm_instance],
    include_contents='none',
    )

# delete_and_log_one_vm_agent = SequentialAgent(
#     name="delete_and_log_one_vm_agent",
#     description="Deletes a single specified VM instance and then logs the deletion event. Use this for deleting one VM.",
#     sub_agents=[delete_vm_instance_agent, log_deletion_agent]
# )

# delete_multiple_ins_loop_agent = LoopAgent(
#     name="delete_multiple_ins_loop_agent",
#     description="""This agent iterates over a list of virtual machines stored in the 'vms_to_delete_list' state variable, deleting and logging each one sequentially. For each VM in the list, it delegates the deletion and logging task to the 'delete_and_log_one_vm_agent'.""",
#     sub_agents=[delete_and_log_one_vm_agent],
#     loop_variable="vms_to_delete_list",
#     item_variable="current_vm_details",
#     max_iterations=10
#     )
# Find your root_agent definition
# In your agent.py file, find the root_agent definition and replace it.

bq_intel_agent = Agent(
  model="gemini-2.0-flash",
  name="bigquery_agent_eval",
  description=(
    "Agent that answers questions about BigQuery data by executing SQL queries"
  ),
  instruction="""
    You are a data analysis agent with access to several BigQuery tools.
    Use the appropriate tools to retrieve BigQuery metadata and execute SQL queries in order to answer the users question.
    Run these queries in the project-id: <PROJECT ID>.
    ALL questions relate to data stored in the <DATASET> dataset.
  """,
  tools=[bigquery_toolset]
)

root_agent = LlmAgent(
    name="finops_optimization_agent",
    model="gemini-2.0-flash", # Or your preferred model
    description=(
        """Agent for Google Cloud finops tasks. Can list, delete, and check CPU utilization of VMs. 
        It can also perform multi-step operations like deleting VMs based on their CPU usage."""
    ),
    instruction=(
        """You are an advanced Google Cloud finops assistant. You can answer questions and execute tasks by calling tools or delegating to sub-agents.

        **Core Capabilities:**
        - Greet the user by delegating to the `greeting_agent`.
        - List running VMs using the `list_vm_instances` tool.
        - Delete one or more VMs by delegating to the `delete_vm_instance_agent`.
        - Check CPU usage for all VMs in a zone using the `call_cpu_utilization_agent` tool.
        - Answer general finops questions using the `search_tool`.

        **IMPORTANT REASONING PROCESS for Deletion by CPU Utilization:**
        When a user asks you to delete VMs based on a condition like "CPU utilization below 30%", you MUST follow this multi-step process:

        1.  **Step 1: Gather Data.** You DO NOT have a tool to directly filter VMs by CPU. Your first action MUST be to call the `call_cpu_utilization_agent` tool with the correct `project_id` and `zone` to get the list of all VMs and their current CPU usage.

        2.  **Step 2: Analyze and Plan.** After you get the text output from `call_cpu_utilization_agent`, you must carefully read it. Parse the text to identify the `Instance ID` of every VM that meets the user's criteria (e.g., CPU percentage is less than 30). Create a list of these target Instance IDs. If no VMs meet the criteria, inform the user and stop.

        3.  **Step 3: Prepare State and Execute Deletion.** After you have identified the `Instance ID`(s) of the VM(s) to delete in Step 2, you MUST perform the following actions:
            a. Create a Python LIST of dictionaries.
            b. For EACH VM to be deleted (whether it's one or many), add a dictionary to the list. Each dictionary MUST have the keys "project_id", "instance_id", and "zone" with their corresponding values.
            c. Save this list to the session state with the key `vms_to_delete_list`.
            d. Delegate the entire deletion task to the `delete_vm_instance_agent`. This is your ONLY way to perform deletions.

        4.  **Step 4: Report to User.** After the deletion agents have finished, provide a clear summary of which VMs were deleted to the user.
        """
    ),
    tools=[
        list_vm_instances, 
        search_tool, 
        call_cpu_utilization_agent
    ],
    sub_agents=[
        delete_vm_instance_agent, 
        greeting_agent,
        bq_intel_agent
    ],
    # Your callbacks remain the same
    before_model_callback=simple_before_model_modifier,
    #after_model_callback=log_interaction_after_model
)

#*************************END: Agents Section**************************************
#*************************START: Agent Common Section**************************************

Initial_state = {
    "user_name": "Robin Varghese",
    "user_preferences": """
        I like to adress the organizational finops challenges is the best and efficient way.
        I use Google cloud services for my work and I usually suggest Google services to my customers.
        My LinkedIn profile can be found at https://www.linkedin.com/in/robinkoikkara/
        """
}

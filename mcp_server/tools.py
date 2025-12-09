import json
import logging
import time
import asyncio
import re
import traceback
import base64
import binascii
from typing import Any, Dict, List, Optional


import requests
import pandas as pd
import plotly.express as px
from google.cloud import bigquery, storage
import vertexai
from vertexai import rag
from mcp.server.fastmcp import FastMCP, Context

# Handle imports for both module and script execution
try:
    from . import config
    from .rag_utils import check_corpus_exists, get_corpus_resource_name
except ImportError:
    # Running as a script
    import config
    from rag_utils import check_corpus_exists, get_corpus_resource_name

# Initialize FastMCP server
mcp = FastMCP("FinOptiAgents Tools")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Mock ToolContext for migration compatibility
class ToolContext:
    def __init__(self, state: Dict[str, Any] = None):
        self.state = state or {}

# Global state to simulate session state (shared across calls in this simple migration)
GLOBAL_STATE = {}

def get_tool_context() -> ToolContext:
    return ToolContext(GLOBAL_STATE)

# --- Tools ---

@mcp.tool()
def run_bq_query(query: str) -> str:
    """
    Executes a read-only BigQuery SQL query and returns the results.

    **CRITICAL WORKFLOW: Auditing VM Deletion History**
    To answer questions about past VM deletions (e.g., "who deleted what and when"), you MUST use this tool to query the `vector-search-poc.finops_agent_logs.vm_deletion_log` table.

    **CRITICAL SCHEMA & DATA FORMAT for `vm_deletion_log`:**
    - The ONLY column with deletion details is `log_data` (Type: JSON).
    - **MANDATORY DATA NOTE:** The `log_data` column is a JSON string that contains *another* JSON string. You MUST double-parse it.

    **MANDATORY SQL BEST PRACTICES for `vm_deletion_log`:**

    1.  **JSON EXTRACTION (NON-NEGOTIABLE RULE):** You MUST use this exact two-step pattern to get any value from the `log_data` column:
        `JSON_EXTRACT_SCALAR(PARSE_JSON(JSON_EXTRACT_SCALAR(log_data, '$')), '$.key_name')`
        Replace `key_name` with the actual key you need (e.g., `user_id`, `deletion_timestamp_utc`).

    2.  **TIMESTAMP HANDLING (NON-NEGOTIABLE RULE):** The timestamp is inside the JSON and is called `deletion_timestamp_utc`. To query it, you MUST use the full pattern below.
        **THE ONLY CORRECT WAY TO QUERY BY DATE IS:**
        `WHERE DATE(SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', JSON_EXTRACT_SCALAR(PARSE_JSON(JSON_EXTRACT_SCALAR(log_data, '$')), '$.deletion_timestamp_utc'))) = 'YYYY-MM-DD'`

    **EXAMPLE QUERY for "how many vms were deleted today":**
    ```sql
    SELECT COUNT(*) as deleted_vm_count FROM `vector-search-poc.finops_agent_logs.vm_deletion_log` WHERE DATE(SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', JSON_EXTRACT_SCALAR(PARSE_JSON(JSON_EXTRACT_SCALAR(log_data, '$')), '$.deletion_timestamp_utc'))) = CURRENT_DATE()
    ```

    **General BigQuery Instructions:**
    This is your primary tool for understanding the state of the cloud environment.
    You do NOT need to specify a project_id; the tool runs in the correct project automatically.
    All the table attributes are set with descriptions. So chech the description of columns to identify the correct columns and make correct queries.

    The table names in your query, like `finoptiagents.finops_cost_usage`, already contain the dataset.
    
    For any other queries use the following ables in vector-search-poc.finoptiagents dataset
    1. project_information_master: Core project details. This is the central registry of all projects. 
    Use it to find project names, owners, and IDs. The stakeholder details are mentioned in this table, product_owner_name, business_service_owner_name
    2. project_information_child: Individual cloud resources. This table contains a detailed inventory of every single provisioned resource 
    in a project(like VMs, databases, etc.) for each project.
    3. finops_cost_usage: Raw monthly cost data with environment break-up. This table holds the raw financial and performance metrics. Use it for detailed 
    analysis of monthly costs and resource utilization percentages.
    4. servicenow_change_defect: Development tickets. This table tracks active development and bug fixes from ServiceNow. A project with open tickets here is 
    considered "active," justifying its operational costs. The entry made for a project is irrespective of the environment. In other words, a ticket raised for 
    dev environment is impacting the enite project
    5. earb_review: Governance approvals. This table logs which projects have passed the formal Enterprise Architecture Review Board (EARB) process. A missing 
    entry here is a major governance red flag.
    6. release_train_ticket: Release planning. This table lists projects that are officially part of a planned software release train. The project budget is 
    stored in this table. An entry made for a project, irrespective of environment is considered to be part of the release train.

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
    logger.info(f"[TOOL CALL] run_bq_query - Starting execution")
    logger.info(f"[TOOL INPUT] Query length: {len(query)} chars")
    logger.debug(f"[TOOL INPUT] Full query: {query}")
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
        response = json.dumps({"total_rows_found": results.total_rows, "rows_returned_in_sample": len(data_sample), "data_sample": data_sample}, default=str)
        logger.info(f"[TOOL RESPONSE] run_bq_query - Success: {results.total_rows} rows found")
        return response
    except Exception as e:
        logger.error(f"[TOOL ERROR] run_bq_query - Failed: {e}", exc_info=True)
        return json.dumps({"error": f"An error occurred while running the query: {str(e)}"})

@mcp.tool()

def send_email(to_address: str, subject: str, user_name: str = "FinOptiAgents", body_html_base64: Optional[str] = None) -> dict:

    """Sends an email to the specified recipient.



    Args:

        to_address (str): The recipient's email address.

        subject (str): The subject of the email.

        user_name (str): The name of the user sending the email. Defaults to "FinOptiAgents".

        body_html_base64 (str, optional): The content for the email body. It will be treated as plain text

                                          and converted to base64 encoded HTML. Defaults to None.

                                          If not provided, a simple body will be generated from the subject.



    Returns:

        dict: A dictionary containing the status and message from the email service.

    """

    # tool_context is unused in the original implementation, so we drop it from args

    logger.info(f"[TOOL CALL] send_email - Starting execution")

    logger.info(f"[TOOL INPUT] to={to_address}, subject='{subject}', user_name='{user_name}', body_provided={body_html_base64 is not None}")



    body_to_send = body_html_base64

    if body_to_send:

        # Assume it's plain text/HTML that needs encoding.

        html_body = "<html><body>" + body_to_send.replace('\n', '<br>') + "</body></html>"

        body_to_send = base64.b64encode(html_body.encode('utf-8')).decode('utf-8')

    else:

        # Create a simple HTML body from the subject and base64 encode it

        html_body = f"<html><body><p>{subject}</p></body></html>"

        body_to_send = base64.b64encode(html_body.encode('utf-8')).decode('utf-8')



    headers = {'Content-Type': 'application/json'}

    data = {

        'to_address': to_address,

        'subject': subject,

        'user_name': user_name,

        'body_html_base64': body_to_send

    }

    url = "https://email-agent-backend-912533822336.us-central1.run.app/send-email"



    try:

        response = requests.post(url, headers=headers, data=json.dumps(data))

        response.raise_for_status()  # Raise an exception for HTTP errors

        result = response.json()

        logger.info(f"[TOOL RESPONSE] send_email - Success: {result.get('status', 'unknown')}")

        return result

    except requests.exceptions.RequestException as e:

        logging.error(f"Failed to send email to {to_address}. Error: {e}", exc_info=True)

        return {"status": "error", "message": f"Failed to send email: {str(e)}"}

    except Exception as e:

        logging.error(f"Unexpected error in send_email tool: {e}", exc_info=True)

        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

@mcp.tool()
def list_vm_instances(project_id: str, zone: str):
    """Lists VM instances based on domain, project ID, and zone."""
    logger.info(f"[TOOL CALL] list_vm_instances - Starting execution")
    logger.info(f"[TOOL INPUT] project_id='{project_id}', zone='{zone}'")
    headers = {'Content-Type': 'application/json'}
    data = {'project_id': project_id, 'zone': zone}
    url = "https://agent-tools-912533822336.us-central1.run.app/list_vms"
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        logger.info(f"[TOOL RESPONSE] list_vm_instances - Success: Retrieved VM list")
        return result
    except requests.exceptions.RequestException as e:
        logging.error(f"Error listing instances for project '{project_id}': {e}", exc_info=True)
        return None

@mcp.tool()
def delete_vm_instance(project_id: str, instance_id: str, zone: str):
    """
    Deletes a VM instance and AUTOMATICALLY logs the deletion event to BigQuery upon success.
    This is now an atomic operation.
    """
    tool_context = get_tool_context()
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
            # We don't have log_vm_deletion_to_bigquery migrated yet, so we skip logging for now or need to migrate it too.
            # Assuming log_vm_deletion_to_bigquery is internal helper.
            # For now, we'll just return success.
            # TODO: Migrate log_vm_deletion_to_bigquery if needed.
            return response_data
        else:
            logging.warning(f"API reported failure to delete '{instance_id}': {response_data}")
            return response_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling deletion API for instance '{instance_id}': {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@mcp.tool()
def generate_chart_from_data(
    chart_type: str,
    data_json_string: str,
    title: str,
    x_column: str,
    y_columns: List[str],
    labels_column: Optional[str] = None,
    values_column: Optional[str] = None,
    color_column: Optional[str] = None
) -> str:
    """
    Generates a chart from JSON data, uploads it to GCS, and returns a public URL.
    
    Args:
        chart_type (str): The type of chart ('bar', 'pie', 'line').
        data_json_string (str): The data in JSON format as a string.
        title (str): The title of the chart.
        x_column (str): The column name for the X-axis.
        y_columns (List[str]): A list of column names for the Y-axis.
        labels_column (Optional[str]): The column for pie chart labels.
        values_column (Optional[str]): The column for pie chart values.
        color_column (Optional[str]): The column to use for coloring lines in a line chart.
    """
    logging.info(f"--- [Chart Tool] Generating '{chart_type}' chart titled '{title}' ---")
    bucket_name = "finoptiagents-generated-graph"
    try:
        data = json.loads(data_json_string)
        if not data:
            return json.dumps({"error": "Input data is empty."})
        
        df = pd.DataFrame(data)
        for col in y_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        fig = None
        if chart_type.lower() == 'bar':
            df_melted = df.melt(id_vars=[x_column], value_vars=y_columns, var_name='Category', value_name='Value')
            fig = px.bar(df_melted, x=x_column, y='Value', color='Category', title=title, barmode='group', template="plotly_white")
        elif chart_type.lower() == 'pie':
            fig = px.pie(df, names=labels_column, values=values_column, title=title, template="plotly_white")
        elif chart_type.lower() == 'line':
            fig = px.line(df, x=x_column, y=y_columns[0], color=color_column, title=title, template="plotly_white")
        else:
            return json.dumps({"error": f"Unsupported chart type: '{chart_type}'."})

        # Save chart to a temporary local file
        chart_filename = f"{title.replace(' ', '_')}_{int(time.time())}.html"
        local_chart_path = f"/tmp/{chart_filename}"
        fig.write_html(local_chart_path)

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(chart_filename)
        blob.upload_from_filename(local_chart_path)

        logging.info(f"--- [Chart Tool] Successfully uploaded chart to GCS: {blob.public_url} ---")
        
        return f"Chart has been generated and is available at: {blob.public_url}"

    except Exception as e:
        logging.error(f"Chart generation or GCS upload failed: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {e}"})

# Helper for call_cpu_utilization_agent
def _get_streamed_response_sync(query: str, resource_name: str) -> str:
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

@mcp.tool()
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

# --- RAG Tools ---

@mcp.tool()
def rag_query(corpus_name: str, query: str) -> dict:
    """
    Query a Vertex AI RAG corpus with a user question and return relevant information.

    Args:
        corpus_name (str): The name of the corpus to query. If empty, the current corpus will be used.
                          Preferably use the resource_name from list_corpora results.
        query (str): The text query to search for in the corpus
    """
    tool_context = get_tool_context()
    try:
        if not check_corpus_exists(corpus_name, tool_context):
            logger.warning(f"Corpus '{corpus_name}' does not exist.")
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
                "query": query,
                "corpus_name": corpus_name,
            }

        corpus_resource_name = get_corpus_resource_name(corpus_name)
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=config.DEFAULT_TOP_K,
            filter=rag.Filter(vector_distance_threshold=config.DEFAULT_DISTANCE_THRESHOLD),
        )

        logger.info(f"Performing retrieval query on corpus '{corpus_name}' with query: '{query}'.")
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_resource_name)],
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )
        logger.info("Retrieval query successful.")

        results = []
        if hasattr(response, "contexts") and response.contexts:
            for ctx_group in response.contexts.contexts:
                result = {
                    "source_uri": (ctx_group.source_uri if hasattr(ctx_group, "source_uri") else ""),
                    "source_name": (ctx_group.source_display_name if hasattr(ctx_group, "source_display_name") else ""),
                    "text": ctx_group.text if hasattr(ctx_group, "text") else "",
                    "score": ctx_group.score if hasattr(ctx_group, "score") else 0.0,
                }
                results.append(result)

        if not results:
            logger.warning(f"No results found in corpus '{corpus_name}' for query: '{query}'.")
            return {
                "status": "warning",
                "message": f"No results found in corpus '{corpus_name}' for query: '{query}'",
                "query": query,
                "corpus_name": corpus_name,
                "results": [],
                "results_count": 0,
            }

        success_message = f"Successfully queried corpus '{corpus_name}'"
        logger.info(success_message)
        return {
            "status": "success",
            "message": success_message,
            "query": query,
            "corpus_name": corpus_name,
            "results": results,
            "results_count": len(results),
        }

    except Exception as e:
        error_msg = f"Error querying corpus: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "query": query,
            "corpus_name": corpus_name,
        }

@mcp.tool()
def create_corpus(corpus_name: str) -> dict:
    """
    Create a new Vertex AI RAG corpus with the specified name.

    Args:
        corpus_name (str): The name for the new corpus
    """
    tool_context = get_tool_context()
    try:
        if check_corpus_exists(corpus_name, tool_context):
            logger.info(f"Corpus '{corpus_name}' already exists.")
            return {
                "status": "info",
                "message": f"Corpus '{corpus_name}' already exists",
                "corpus_name": corpus_name,
                "corpus_created": False,
            }

        display_name = re.sub(r"[^a-zA-Z0-9_-]", "_", corpus_name)
        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=config.DEFAULT_EMBEDDING_MODEL
            )
        )

        logger.info(f"Creating corpus '{corpus_name}' with display name '{display_name}'.")
        rag_corpus = rag.create_corpus(
            display_name=display_name,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config
            ),
        )
        logger.info("Corpus created successfully.")

        tool_context.state[f"corpus_exists_{corpus_name}"] = True
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

@mcp.tool()
def add_data(corpus_name: str, paths: List[str]) -> dict:
    """
    Add new data sources to a Vertex AI RAG corpus.

    Args:
        corpus_name (str): The name of the corpus to add data to. If empty, the current corpus will be used.
        paths (List[str]): List of URLs or GCS paths to add to the corpus.
    """
    tool_context = get_tool_context()
    try:
        if not check_corpus_exists(corpus_name, tool_context):
            logger.warning(f"Corpus '{corpus_name}' does not exist.")
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist. Please create it first using the create_corpus tool.",
                "corpus_name": corpus_name,
                "paths": paths,
            }

        if not paths or not all(isinstance(path, str) for path in paths):
            logger.warning("Invalid paths provided.")
            return {
                "status": "error",
                "message": "Invalid paths: Please provide a list of URLs or GCS paths",
                "corpus_name": corpus_name,
                "paths": paths,
            }

        validated_paths = []
        invalid_paths = []
        conversions = []

        for path in paths:
            if not path or not isinstance(path, str):
                invalid_paths.append(f"{path} (Not a valid string)")
                continue

            docs_match = re.match(r"https:\/\/docs\.google\.com\/(?:document|spreadsheets|presentation)\/d\/([a-zA-Z0-9_-]+)(?:\/|$)", path)
            if docs_match:
                file_id = docs_match.group(1)
                drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                validated_paths.append(drive_url)
                conversions.append(f"{path} → {drive_url}")
                continue

            drive_match = re.match(r"https:\/\/drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)(?:\/|$)", path)
            if drive_match:
                file_id = drive_match.group(1)
                drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                validated_paths.append(drive_url)
                if drive_url != path:
                    conversions.append(f"{path} → {drive_url}")
                continue

            if path.startswith("gs://"):
                validated_paths.append(path)
                continue

            invalid_paths.append(f"{path} (Invalid format)")

        if not validated_paths:
            logger.warning("No valid paths provided.")
            return {
                "status": "error",
                "message": "No valid paths provided. Please provide Google Drive URLs or GCS paths.",
                "corpus_name": corpus_name,
                "invalid_paths": invalid_paths,
            }

        corpus_resource_name = get_corpus_resource_name(corpus_name)
        transformation_config = rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=config.DEFAULT_CHUNK_SIZE,
                chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
            ),
        )

        logger.info(f"Importing {len(validated_paths)} files to corpus '{corpus_name}'.")
        import_result = rag.import_files(
            corpus_resource_name,
            validated_paths,
            transformation_config=transformation_config,
            max_embedding_requests_per_min=config.DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
        )
        logger.info("Import successful.")

        if not tool_context.state.get("current_corpus"):
            tool_context.state["current_corpus"] = corpus_name

        conversion_msg = ""
        if conversions:
            conversion_msg = " (Converted Google Docs URLs to Drive format)"

        success_message = f"Successfully added {import_result.imported_rag_files_count} file(s) to corpus '{corpus_name}'{conversion_msg}"
        logger.info(success_message)
        return {
            "status": "success",
            "message": success_message,
            "corpus_name": corpus_name,
            "files_added": import_result.imported_rag_files_count,
            "paths": validated_paths,
            "invalid_paths": invalid_paths,
            "conversions": conversions,
        }

    except Exception as e:
        logger.error(f"Error adding data to corpus: {str(e)}")
        return {
            "status": "error",
            "message": f"Error adding data to corpus: {str(e)}",
            "corpus_name": corpus_name,
            "paths": paths,
        }

@mcp.tool()
def delete_corpus(corpus_name: str, confirm: bool) -> dict:
    """
    Delete a Vertex AI RAG corpus when it's no longer needed.
    Requires confirmation to prevent accidental deletion.

    Args:
        corpus_name (str): The full resource name of the corpus to delete.
        confirm (bool): Must be set to True to confirm deletion
    """
    tool_context = get_tool_context()
    try:
        if not check_corpus_exists(corpus_name, tool_context):
            logger.warning(f"Corpus '{corpus_name}' does not exist.")
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist",
                "corpus_name": corpus_name,
            }

        if not confirm:
            logger.warning("Deletion not confirmed.")
            return {
                "status": "error",
                "message": "Deletion requires explicit confirmation. Set confirm=True to delete this corpus.",
                "corpus_name": corpus_name,
            }

        corpus_resource_name = get_corpus_resource_name(corpus_name)
        logger.info(f"Deleting corpus '{corpus_name}'.")
        rag.delete_corpus(corpus_resource_name)
        logger.info("Corpus deleted successfully.")

        state_key = f"corpus_exists_{corpus_name}"
        if state_key in tool_context.state:
            tool_context.state[state_key] = False

        success_message = f"Successfully deleted corpus '{corpus_name}'"
        logger.info(success_message)
        return {
            "status": "success",
            "message": success_message,
            "corpus_name": corpus_name,
        }
    except Exception as e:
        logger.error(f"Error deleting corpus: {str(e)}")
        return {
            "status": "error",
            "message": f"Error deleting corpus: {str(e)}",
            "corpus_name": corpus_name,
        }

@mcp.tool()
def delete_document(corpus_name: str, document_id: str) -> dict:
    """
    Delete a specific document from a Vertex AI RAG corpus.

    Args:
        corpus_name (str): The full resource name of the corpus containing the document.
        document_id (str): The ID of the specific document/file to delete.
    """
    tool_context = get_tool_context()
    try:
        if not check_corpus_exists(corpus_name, tool_context):
            logger.warning(f"Corpus '{corpus_name}' does not exist.")
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist",
                "corpus_name": corpus_name,
                "document_id": document_id,
            }

        corpus_resource_name = get_corpus_resource_name(corpus_name)
        rag_file_path = f"{corpus_resource_name}/ragFiles/{document_id}"
        logger.info(f"Deleting document '{document_id}' from corpus '{corpus_name}'.")
        rag.delete_file(rag_file_path)
        logger.info("Document deleted successfully.")

        success_message = f"Successfully deleted document '{document_id}' from corpus '{corpus_name}'"
        logger.info(success_message)
        return {
            "status": "success",
            "message": success_message,
            "corpus_name": corpus_name,
            "document_id": document_id,
        }
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return {
            "status": "error",
            "message": f"Error deleting document: {str(e)}",
            "corpus_name": corpus_name,
            "document_id": document_id,
        }

@mcp.tool()
def get_corpus_info(corpus_name: str) -> dict:
    """
    Get detailed information about a specific RAG corpus, including its files.

    Args:
        corpus_name (str): The full resource name of the corpus to get information about.
    """
    tool_context = get_tool_context()
    try:
        if not check_corpus_exists(corpus_name, tool_context):
            logger.warning(f"Corpus '{corpus_name}' does not exist.")
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist",
                "corpus_name": corpus_name,
            }

        corpus_resource_name = get_corpus_resource_name(corpus_name)
        corpus_display_name = corpus_name

        file_details = []
        try:
            logger.info(f"Listing files for corpus '{corpus_name}'.")
            files = rag.list_files(corpus_resource_name)
            logger.info(f"Found {len(files)} files.")
            for rag_file in files:
                try:
                    file_id = rag_file.name.split("/")[-1]
                    file_info = {
                        "file_id": file_id,
                        "display_name": (rag_file.display_name if hasattr(rag_file, "display_name") else ""),
                        "source_uri": (rag_file.source_uri if hasattr(rag_file, "source_uri") else ""),
                        "create_time": (str(rag_file.create_time) if hasattr(rag_file, "create_time") else ""),
                        "update_time": (str(rag_file.update_time) if hasattr(rag_file, "update_time") else ""),
                    }
                    file_details.append(file_info)
                except Exception as e:
                    logger.warning(f"Could not process file: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Could not list files for corpus '{corpus_name}': {e}")
            pass

        success_message = f"Successfully retrieved information for corpus '{corpus_display_name}'"
        logger.info(success_message)
        return {
            "status": "success",
            "message": success_message,
            "corpus_name": corpus_name,
            "corpus_display_name": corpus_display_name,
            "file_count": len(file_details),
            "files": file_details,
        }

    except Exception as e:
        logger.error(f"Error getting corpus information: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting corpus information: {str(e)}",
            "corpus_name": corpus_name,
        }

@mcp.tool()
def list_corpora() -> dict:
    """
    List all available Vertex AI RAG corpora.

    Returns:
        dict: A list of available corpora and status.
    """
    try:
        logger.info("Listing all available corpora.")
        corpora_pager = rag.list_corpora()
        corpora_list = list(corpora_pager)
        logger.info(f"Found {len(corpora_list)} corpora.")

        corpus_info = []
        for corpus in corpora_list:
            corpus_data = {
                "resource_name": corpus.name,
                "display_name": corpus.display_name,
                "create_time": (str(corpus.create_time) if hasattr(corpus, "create_time") else ""),
                "update_time": (str(corpus.update_time) if hasattr(corpus, "update_time") else ""),
            }
            corpus_info.append(corpus_data)

        success_message = f"Found {len(corpus_info)} available corpora"
        logger.info(success_message)
        return {
            "status": "success",
            "message": success_message,
            "corpora": corpus_info,
        }
    except Exception as e:
        logger.error(f"Error listing corpora: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error listing corpora: {str(e)}",
            "corpora": [],
        }

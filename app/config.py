"""
Centralized configuration settings for the FinOps Agent.

This file loads all necessary settings from environment variables and Google Secret Manager.
It also initializes Google services like Vertex AI and the Generative AI client.
All other modules in the application should import their configuration from this file.
"""

import os
import google.auth
from google.cloud import secretmanager
from google.cloud import resourcemanager_v3
import logging
from google.api_core import exceptions
import vertexai
import google.genai as genai

# =======================================================================================
# 1. Centralized Logging Configuration
# =======================================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =======================================================================================
# 2. Load environment variables from .env file
# =======================================================================================


# =======================================================================================
# 3. Static RAG Agent Constants
# =======================================================================================
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3
DEFAULT_DISTANCE_THRESHOLD = 0.5
DEFAULT_EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"
DEFAULT_EMBEDDING_REQUESTS_PER_MIN = 1000
FINOPS_CORPUS_DISPLAY_NAME = "finops_design_documents_corpus"


# =======================================================================================
# 4. Secret Manager and Project Resolution Helper Functions
# =======================================================================================
def _get_secret_value(project_id: str, secret_id: str, client: secretmanager.SecretManagerServiceClient) -> str | None:
    """Helper function to fetch a single secret from Secret Manager."""
    if not project_id:
        return None
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    try:
        response = client.access_secret_version(request={"name": name})
        value = response.payload.data.decode("UTF-8")
        logging.info(f"Successfully fetched secret: '{secret_id}'")
        return value
    except exceptions.NotFound:
        logging.warning(f"Secret '{secret_id}' not found in project '{project_id}'.")
        return None
    except Exception as e:
        logging.warning(f"Could not fetch secret '{secret_id}': {e}")
        return None

def _resolve_project_id_from_number(project_number: str) -> str | None:
    """
    Given a project number, resolves it to the project ID string.
    Returns None if resolution fails.
    """
    try:
        logging.info(f"Attempting to resolve project ID from project number: {project_number}...")
        client = resourcemanager_v3.ProjectsClient()
        project_path = f"projects/{project_number}"
        project = client.get_project(name=project_path)
        project_id = project.project_id
        logging.info(f"Successfully resolved project number '{project_number}' to project ID: '{project_id}'")
        return project_id
    except Exception as e:
        logging.error(f"Failed to resolve project ID from number '{project_number}': {e}", exc_info=True)
        return None


# =======================================================================================
# 5. Core Configuration Loading
# =======================================================================================
print("--- Loading configuration ---")
_secret_client = secretmanager.SecretManagerServiceClient()

# Determine the Project ID using a robust fallback mechanism
_initial_project_identifier = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not _initial_project_identifier:
    try:
        _, _initial_project_identifier = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        _initial_project_identifier = None

# First, try to get the project ID from a specific secret, otherwise use the discovered one.
_project_identifier_from_secret = _get_secret_value(_initial_project_identifier, "google-project-id", _secret_client)
_final_project_identifier = _project_identifier_from_secret or _initial_project_identifier


# If the identifier is a number, resolve it to the project ID string
if _final_project_identifier and _final_project_identifier.isdigit():
    logging.info(f"Project identifier '{_final_project_identifier}' appears to be a project number.")
    GOOGLE_PROJECT_ID = _resolve_project_id_from_number(_final_project_identifier)
else:
    GOOGLE_PROJECT_ID = _final_project_identifier


if not GOOGLE_PROJECT_ID:
    raise ValueError("FATAL: Could not determine Google Cloud Project ID. Please set GOOGLE_CLOUD_PROJECT or the 'google-project-id' secret.")
logging.info(f"Using Project ID: {GOOGLE_PROJECT_ID}")

# --- ADDED FOR BACKWARDS COMPATIBILITY ---
PROJECT_ID = GOOGLE_PROJECT_ID


# Helper to fetch other secrets using the now-confirmed Project ID
def _fetch_config(secret_id: str) -> str | None:
    return _get_secret_value(GOOGLE_PROJECT_ID, secret_id, _secret_client)

# Load all other configuration values
GOOGLE_API_KEY = _fetch_config("google-api-key")
# Use GOOGLE_ZONE secret, fall back to GOOGLE_CLOUD_LOCATION env var
GOOGLE_ZONE = _fetch_config("google-zone") or os.environ.get("GOOGLE_CLOUD_LOCATION")
REMOTE_CPU_AGENT_RESOURCE_NAME = _fetch_config("remote-cpu-agent-resource-name")
REMOTE_RAG_AGENT_RESOURCE_NAME = _fetch_config("remote-rag-agent-resource-name")
EARB_DESIGNDOCS = _fetch_config("rag-earb-designdocs")  # e.g., "gs://my-finops-design-docs-bucket"

# --- ADDED FOR BACKWARDS COMPATIBILITY ---
LOCATION = GOOGLE_ZONE
RAG_REGION = _fetch_config("rag-engine-location")
BIGQUERY_DATASET_ID = _fetch_config("bigquery-dataset-id") or "finoptiagents"
BIGQUERYAGENTANALYTICSPLUGIN_TABLE_ID = _fetch_config("bigquery-agent-analytics-table-id") or "agent_analytics_log"
FINOPTIAGENTS_LLM = _fetch_config("finoptiagents-llm") or "gemini-3-flash-preview"

logging.info("--- Configuration loading complete. ---")


# =======================================================================================
# 6. Initialize Google Cloud Services
# =======================================================================================
if GOOGLE_PROJECT_ID and RAG_REGION:
    # Vertex AI SDK requires a region (e.g., "us-central1", "us-east4"), not a zone (e.g., "us-central1-a")
    #google_region = "-".join(RAG_REGION.split("-")[:-1])
    try:
        logging.info(f"Initializing Vertex AI for project '{GOOGLE_PROJECT_ID}' in region '{RAG_REGION}'...")
        vertexai.init(project=GOOGLE_PROJECT_ID, location=RAG_REGION)
        logging.info("Vertex AI initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)

if GOOGLE_API_KEY:
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    logging.info("Google API Key set as environment variable.")
else:
    logging.warning("GOOGLE_API_KEY not found. Some Generative AI features may not be available.")

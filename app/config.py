"""
Centralized configuration settings for the FinOps Agent.

This file loads all necessary settings from environment variables and Google Secret Manager.
It also initializes Google services like Vertex AI and the Generative AI client.
All other modules in the application should import their configuration from this file.
"""

import os
import google.auth
from google.cloud import secretmanager
import logging
from google.api_core import exceptions
import vertexai
import google.generativeai as genai

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
# 4. Secret Manager Helper Functions
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


# =======================================================================================
# 5. Core Configuration Loading
# =======================================================================================
print("--- Loading configuration ---")
_secret_client = secretmanager.SecretManagerServiceClient()

# Determine the Project ID using a robust fallback mechanism
_initial_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not _initial_project_id:
    try:
        _, _initial_project_id = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        _initial_project_id = None

# First, try to get the project ID from a specific secret, otherwise use the discovered one.
GOOGLE_PROJECT_ID = _get_secret_value(_initial_project_id, "google-project-id", _secret_client) or _initial_project_id

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
EARB_DESIGNDOCS = _fetch_config("earb-designdocs")  # e.g., "gs://my-finops-design-docs-bucket"

# --- ADDED FOR BACKWARDS COMPATIBILITY ---
LOCATION = GOOGLE_ZONE
RAG_REGION = "us-east4"

logging.info("--- Configuration loading complete. ---")


# =======================================================================================
# 6. Initialize Google Cloud Services
# =======================================================================================
if GOOGLE_PROJECT_ID and GOOGLE_ZONE:
    # Vertex AI SDK requires a region (e.g., "us-central1"), not a zone (e.g., "us-central1-a")
    google_region = "-".join(GOOGLE_ZONE.split("-")[:-1])
    try:
        logging.info(f"Initializing Vertex AI for project '{GOOGLE_PROJECT_ID}' in region '{google_region}'...")
        vertexai.init(project=GOOGLE_PROJECT_ID, location=google_region)
        logging.info("Vertex AI initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("Google Generative AI client configured successfully.")
    except Exception as e:
        logging.error(f"Failed to configure Generative AI client: {e}", exc_info=True)
else:
    logging.warning("GOOGLE_API_KEY not found. Some Generative AI features may not be available.")
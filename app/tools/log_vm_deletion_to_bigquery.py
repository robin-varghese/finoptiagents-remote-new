import datetime
import json
import logging

from google.adk.tools import ToolContext
from google.cloud import bigquery

from .. import config
from ..utils.embeddings import generate_combined_embedding

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

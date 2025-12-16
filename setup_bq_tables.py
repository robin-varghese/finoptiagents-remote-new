from google.cloud import bigquery
from google.api_core.exceptions import NotFound, Conflict
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset_and_tables(project_id, dataset_id="finoptiagents"):
    client = bigquery.Client(project=project_id)
    dataset_ref = f"{project_id}.{dataset_id}"

    # 1. Create Dataset
    try:
        client.get_dataset(dataset_ref)
        logging.info(f"Dataset {dataset_ref} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"  # Adjust location if needed (e.g., "europe-west2")
        # Default to US or maybe check where the previous one was. 
        # Safer to stick to US/multi-region or use default.
        # Given user is in US/Europe mix, multi-region US is often default, 
        # but let's check existing dataset location if possible? 
        # For now, I'll default to US as it's standard, or I can remove location to let BQ decide.
        # Better: let BQ decide by not setting it explicitly if possible, or pick "US".
        client.create_dataset(dataset, timeout=30)
        logging.info(f"Created dataset {dataset_ref}")

    # 2. Create cloud_operations_log Table
    table_id = f"{dataset_ref}.cloud_operations_log"
    schema = [
        bigquery.SchemaField("operation_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("actor", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("action_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("resource_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("details", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("operation_source", "STRING", mode="NULLABLE")
    ]
    formatted_table_id = f"{project_id}.{dataset_id}.cloud_operations_log"
    create_table_if_not_exists(client, formatted_table_id, schema)

    # 3. Create cost_savings_log Table
    table_id = f"{dataset_ref}.cost_savings_log"
    schema = [
        bigquery.SchemaField("operation_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("savings_amount", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("currency", "STRING", mode="REQUIRED"), # e.g., 'USD'
        bigquery.SchemaField("duration", "STRING", mode="NULLABLE"), # e.g., 'monthly'
        bigquery.SchemaField("recommendation_id", "STRING", mode="NULLABLE")
    ]
    formatted_table_id = f"{project_id}.{dataset_id}.cost_savings_log"
    create_table_if_not_exists(client, formatted_table_id, schema)

def create_table_if_not_exists(client, table_id, schema):
    try:
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        logging.info(f"Created table {table_id}")
    except Conflict:
        logging.info(f"Table {table_id} already exists.")
    except Exception as e:
        logging.error(f"Failed to create table {table_id}: {e}")

if __name__ == "__main__":
    # Ensure GOOGLE_USAGE_PROJECT_ID or project_id is available
    project_id = os.environ.get("GOOGLE_PROJECT_ID", "vector-search-poc")
    if not project_id:
        logging.error("GOOGLE_PROJECT_ID environment variable not set. Using default 'vector-search-poc'")
        project_id = "vector-search-poc"
        
    logging.info(f"Setting up BigQuery tables for project: {project_id}")
    create_dataset_and_tables(project_id)


import logging
from google.cloud import bigquery

# TODO: Verify this schema against the official BigQueryAnalyticsPlugin documentation
# The official documentation can be found at: https://google.github.io/adk-docs/tools/google-cloud/bigquery-agent-analytics/

# Configuration
PROJECT_ID = "vector-search-poc"  # Your GCP project ID
DATASET_ID = "finoptiagents"  # Your BigQuery dataset
TABLE_ID = "agent_analytics_log"    # The table to store analytics

def create_analytics_table():
    """
    Creates the BigQuery table for agent analytics if it doesn't already exist.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        client = bigquery.Client(project=PROJECT_ID)
        dataset_ref = client.dataset(DATASET_ID)
        table_ref = dataset_ref.table(TABLE_ID)

        # Check if the table already exists
        try:
            client.get_table(table_ref)
            logger.info(f"Table {PROJECT_ID}.{DATASET_ID}.{TABLE_ID} already exists.")
            return
        except Exception:
            logger.info(f"Table {PROJECT_ID}.{DATASET_ID}.{TABLE_ID} not found. Creating it...")

        # Define the schema
        schema = [
            bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("invocation_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("event_author", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("agent_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("event_content", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("is_final_response", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
        ]

        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        logger.info(f"Successfully created table {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")

    except Exception as e:
        logger.error(f"Failed to create BigQuery table: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    create_analytics_table()

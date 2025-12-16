
import os
import sys
import logging
from google.cloud import bigquery
import asyncio

# Setup paths
sys.path.append(os.getcwd())

# Mock config if needed, or rely on env vars
os.environ["GOOGLE_PROJECT_ID"] = "vector-search-poc"
os.environ["GOOGLE_ZONE"] = "us-central1-a"

from mcp_server.tools import log_savings_impact, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_insertion(operation_id):
    client = bigquery.Client(project=config.GOOGLE_PROJECT_ID)
    query = f"""
        SELECT * FROM `{config.GOOGLE_PROJECT_ID}.finoptiagents.cost_savings_log`
        WHERE operation_id = '{operation_id}'
    """
    results = client.query(query).result()
    rows = list(results)
    if len(rows) > 0:
        logger.info(f"✅ Verification Success: Found {len(rows)} row(s) for operation_id {operation_id}")
        for row in rows:
            logger.info(f"   - Amount: {row.savings_amount} {row.currency}")
    else:
        logger.error(f"❌ Verification Failed: No rows found for operation_id {operation_id}")

def main():
    logger.info("Starting Cost Saving Log Test...")
    
    # Test Data
    op_id = "test-log-savings-verification-script"
    amount = 123.45
    
    # Call the tool
    logger.info(f"Calling log_savings_impact with id={op_id}, amount={amount}...")
    result = log_savings_impact(operation_id=op_id, savings_amount=amount, currency="USD")
    logger.info(f"Tool Result: {result}")
    
    # Verify
    if "Successfully logged" in result:
        verify_insertion(op_id)
    else:
        logger.error("❌ Tool execution failed.")

if __name__ == "__main__":
    main()

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.tools.gcloud_mcp_tools import run_gcloud_command
from app.tools.monitoring_mcp_tools import list_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)

async def verify_gcloud():
    print("\n--- Verifying GCloud MCP ---")
    try:
        # Try a simple command: gcloud config list
        result = await run_gcloud_command(args=['config', 'list'])
        print("Result:")
        print(result)
        if "is active" in result or "account =" in result:
            print("✅ GCloud MCP verification passed.")
        else:
            print("⚠️ GCloud MCP output unexpected (but might be valid).")
    except Exception as e:
        print(f"❌ GCloud MCP verification failed: {e}")

async def verify_monitoring():
    print("\n--- Verifying Monitoring MCP ---")
    try:
        # Try listing metrics (requires project_id)
        # We'll try to get project_id from gcloud config first if possible, or use a dummy one if not critical for connection test
        # But list_metrics needs a project_id.
        # Let's try to get it from environment or config.
        from app.config import GOOGLE_PROJECT_ID
        if not GOOGLE_PROJECT_ID:
            print("⚠️ GOOGLE_PROJECT_ID not found, skipping Monitoring verification.")
            return

        print(f"Listing metrics for project: {GOOGLE_PROJECT_ID}")
        result = await list_metrics(project_id=GOOGLE_PROJECT_ID)
        print("Result snippet:")
        print(result[:200] + "..." if len(result) > 200 else result)
        
        if "metricDescriptors" in result or "metrics" in result:
             print("✅ Monitoring MCP verification passed.")
        else:
             print("⚠️ Monitoring MCP output unexpected.")

    except Exception as e:
        print(f"❌ Monitoring MCP verification failed: {e}")

async def main():
    await verify_gcloud()
    await verify_monitoring()

if __name__ == "__main__":
    asyncio.run(main())

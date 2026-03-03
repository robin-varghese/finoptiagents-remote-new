import sys
import os
import logging
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MOCK REMOVED ---
# Allowing native mcp import
pass
try:
    # Ensure current directory is in path
    sys.path.append(os.getcwd())
    from mcp_server.tools import scan_cost_recommendations
except ImportError as e:
    print(f"CRITICAL: Failed to import tools: {e}")
    sys.exit(1)

# --- EXECUTE SCAN ---
PROJECT_ID = "vector-search-poc"

if __name__ == "__main__":
    print(f"🚀 Starting Cost Scan for project: {PROJECT_ID}")
    
    try:
        # Run the tool
        # Note: This will execute REAL gcloud commands via subprocess as defined in the tool
        result_json = scan_cost_recommendations(project_id=PROJECT_ID)
        
        print("\n" + "="*40)
        print("SCAN COMPLETED - RAW OUTPUT")
        print("="*40)
        print(result_json)
        print("="*40)
        
    except Exception as e:
        print(f"❌ Error executing scan: {e}")
        import traceback
        traceback.print_exc()

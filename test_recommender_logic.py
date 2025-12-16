import logging
import sys
from unittest.mock import MagicMock, patch

# Remove global 'mcp' mock to avoid breaking real imports in dependencies
# sys.modules["mcp"] = MagicMock() 

# Mock ONLY the specific module used by tools.py
mock_fastmcp_module = MagicMock()
sys.modules["mcp.server.fastmcp"] = mock_fastmcp_module

# Configure FastMCP class mock
mock_fastmcp_class = MagicMock()
def tool_decorator(*args, **kwargs):
    def wrapper(func):
        return func
    return wrapper
mock_fastmcp_class.return_value.tool.side_effect = tool_decorator

# Assign the mock class to the module
mock_fastmcp_module.FastMCP = mock_fastmcp_class
mock_fastmcp_module.Context = MagicMock()

# Mock config
sys.modules["config"] = MagicMock()
sys.modules["config"].GOOGLE_PROJECT_ID = "test-project"

# Apply Mocks via import
from mcp_server.tools import scan_cost_recommendations

def test_scan_logic():
    print("Testing scan_cost_recommendations logic...")

    # Mock subprocess to capture commands
    with patch("subprocess.run") as mock_run:
        # Mock responses for discovery commands
        def side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)
            print(f"DEBUG: Mock called with: {cmd_str}")
            mock_res = MagicMock()
            mock_res.returncode = 0
            
            if "instances list" in cmd_str:
                mock_res.stdout = "us-central1-a\nus-central1-b"
            elif "addresses list" in cmd_str:
                mock_res.stdout = "us-west1" # Returns region directly
            elif "recommendations list" in cmd_str:
                mock_res.stdout = "[]" 
            return mock_res
            
        mock_run.side_effect = side_effect

        # Run the tool
        scan_cost_recommendations(project_id="test-project", zone="-")

        # Analyze calling args
        calls = mock_run.call_args_list
        commands = [" ".join(call[0][0]) for call in calls]
        
        print(f"DEBUG: Total captured commands: {len(commands)}")
        for i, c in enumerate(commands):
            print(f"DEBUG: Cmd {i}: {c}")
        
        # Verify Global Recommender ran once (global)
        global_check = any("google.compute.image.IdleResourceRecommender" in c and "--location=global" in c for c in commands)
        print(f"Global Recommender (Image) checked strictly globally: {global_check}")

        # Verify Zonal Recommender ran for zones
        zonal_check = any("google.compute.instance.IdleResourceRecommender" in c and "--location=us-central1-a" in c for c in commands)
        print(f"Zonal Recommender (Instance) checked in us-central1-a: {zonal_check}")
        
        # Verify Regional Recommender ran for inferred regions
        # us-central1 (from zones) and us-west1 (from addresses)
        regional_check = any("google.cloudsql.instance.IdleRecommender" in c and "--location=us-central1" in c for c in commands)
        print(f"Regional Recommender (SQL) checked in us-central1: {regional_check}")
        
        if global_check and zonal_check and regional_check:
             print("SUCCESS: Logic correctly routes recommenders to locations.")
        else:
             print("FAILURE: Routing logic mismatch.")
             for c in commands:
                 print(c)

if __name__ == "__main__":
    test_scan_logic()

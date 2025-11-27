
import asyncio
import os
import sys
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
mcp_server_script = os.path.join(project_root, "mcp_server", "main.py")

# --- MONKEYPATCHES (Required for connection) ---
try:
    import google.adk.tools.mcp_tool.mcp_session_manager as mcp_session_manager_module
    from datetime import timedelta
    
    OriginalClientSession = mcp_session_manager_module.ClientSession
    
    class PatchedClientSession(OriginalClientSession):
        def __init__(self, *args, **kwargs):
            # Force a 60-second timeout
            kwargs['read_timeout_seconds'] = timedelta(seconds=60)
            super().__init__(*args, **kwargs)
    
    mcp_session_manager_module.ClientSession = PatchedClientSession

    import google.adk.tools.mcp_tool.mcp_toolset as mcp_toolset_module
    OriginalMcpToolset = mcp_toolset_module.McpToolset
    original_get_tools = OriginalMcpToolset.get_tools

    async def patched_get_tools(self, *args, **kwargs):
        if not hasattr(self._connection_params, 'timeout'):
            self._connection_params.timeout = 60.0
        elif self._connection_params.timeout is None or self._connection_params.timeout < 60.0:
            self._connection_params.timeout = 60.0
        return await original_get_tools(self, *args, **kwargs)

    OriginalMcpToolset.get_tools = patched_get_tools

except ImportError as e:
    logger.error(f"Failed to import module for patching: {e}")
    sys.exit(1)

# --- Import ADK Components ---
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters
from google.adk.agents.readonly_context import ReadonlyContext

async def test_tool_call():
    logger.info("Starting tool call test...")
    
    # Define a custom parameters class to inject timeout
    class TimeoutStdioServerParameters(StdioServerParameters):
        timeout: float | None = 60.0

    # Create Toolset
    toolset = McpToolset(
        connection_params=TimeoutStdioServerParameters(
            command=sys.executable,
            args=[mcp_server_script],
            cwd=project_root,
            env={
                **os.environ,
                "PYTHONPATH": project_root
            }
        )
    )
    
    try:
        # Mock context
        class MockContext(ReadonlyContext):
            def __init__(self):
                pass
        
        # 1. Get Tools (to initialize session)
        logger.info("Initializing session and fetching tools...")
        tools = await toolset.get_tools(MockContext())
        
        # 2. Find the target tool
        target_tool_name = "list_vm_instances"
        target_tool = next((t for t in tools if t.name == target_tool_name), None)
        
        if not target_tool:
            logger.error(f"Tool '{target_tool_name}' not found!")
            return

        # 3. Call the tool
        logger.info(f"Calling tool '{target_tool_name}'...")
        input_args = {
            "project_id": "vector-search-poc",
            "zone": "us-east1-a"
        }
        logger.info(f"Input: {input_args}")
        
        # Note: McpTool.run_async expects a context and input dictionary
        # We need to check how McpTool.run_async is implemented.
        # It usually takes (context, **kwargs) or (context, input_dict) depending on ADK version.
        # Based on previous debugging, we should pass arguments as kwargs or a dict?
        # Let's try passing as kwargs which is standard for ADK tools.
        
        # Wait, McpTool.run_async signature in ADK might be:
        # async def run_async(self, context: ReadonlyContext, **kwargs) -> Any:
        
        result = await target_tool.run_async(MockContext(), **input_args)
        
        logger.info("Tool execution completed.")
        logger.info(f"Result: {result}")
            
    except Exception as e:
        logger.error("FAILURE! Exception occurred:", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_tool_call())

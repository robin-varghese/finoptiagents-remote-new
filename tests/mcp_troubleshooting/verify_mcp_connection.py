
import asyncio
import os
import sys
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
mcp_server_script = os.path.join(project_root, "mcp_server", "main.py")

logger.info(f"Project Root: {project_root}")
logger.info(f"MCP Server Script: {mcp_server_script}")

# --- MONKEYPATCH REPRODUCTION ---
logger.info("Applying Monkeypatches...")
try:
    import google.adk.tools.mcp_tool.mcp_session_manager as mcp_session_manager_module
    from datetime import timedelta
    
    OriginalClientSession = mcp_session_manager_module.ClientSession
    
    class PatchedClientSession(OriginalClientSession):
        def __init__(self, *args, **kwargs):
            logger.info(f"PatchedClientSession initialized! Args: {args}, Kwargs: {kwargs}")
            # Force a 60-second timeout, OVERRIDING any existing value
            kwargs['read_timeout_seconds'] = timedelta(seconds=60)
            super().__init__(*args, **kwargs)
    
    mcp_session_manager_module.ClientSession = PatchedClientSession
    logger.info("ClientSession Monkeypatch applied.")

    # --- MONKEYPATCH 2: Fix for McpToolset.get_tools Timeout ---
    import google.adk.tools.mcp_tool.mcp_toolset as mcp_toolset_module
    import asyncio

    OriginalMcpToolset = mcp_toolset_module.McpToolset
    original_get_tools = OriginalMcpToolset.get_tools

    async def patched_get_tools(self, *args, **kwargs):
        if not hasattr(self._connection_params, 'timeout'):
            self._connection_params.timeout = 60.0
        elif self._connection_params.timeout is None or self._connection_params.timeout < 60.0:
            self._connection_params.timeout = 60.0
        return await original_get_tools(self, *args, **kwargs)

    OriginalMcpToolset.get_tools = patched_get_tools
    logger.info("McpToolset.get_tools Monkeypatch applied.")

except ImportError as e:
    logger.error(f"Failed to import module for patching: {e}")
    sys.exit(1)

# --- Import ADK Components ---
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters

async def verify_connection():
    logger.info("Starting connection verification...")
    
    # Define a custom parameters class to inject timeout (matching app/agent.py)
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
    
    logger.info("McpToolset created. Attempting to get tools...")
    
    try:
        # We need a mock context, or just pass None if allowed (ADK usually requires context)
        # But McpToolset.get_tools() might work without it if we call it directly?
        # Actually McpToolset.get_tools() is what we want to test.
        
        # However, get_tools() is usually called by the agent. 
        # Let's look at McpToolset.get_tools signature.
        # It takes 'context'.
        
        from google.adk.agents.readonly_context import ReadonlyContext
        
        # Mock context
        class MockContext(ReadonlyContext):
            def __init__(self):
                pass
            
        tools = await toolset.get_tools(MockContext())
        
        logger.info("SUCCESS! Tools retrieved:")
        for tool in tools:
            logger.info(f" - {tool.name}")
            
        # --- TEST SPECIFIC TOOL CALL ---
        target_tool_name = "list_vm_instances"
        target_tool = next((t for t in tools if t.name == target_tool_name), None)
        
        if target_tool:
            logger.info(f"\n--- Testing Tool: {target_tool_name} ---")
            input_args = {
                "project_id": "vector-search-poc",
                "zone": "us-east1-a"
            }
            logger.info(f"Input: {input_args}")
            try:
                # McpTool.run_async signature: (self, *, args: dict, tool_context: ToolContext)
                result = await target_tool.run_async(args=input_args, tool_context=MockContext())
                logger.info("Tool execution completed.")
                logger.info(f"Result: {result}")
            except Exception as e:
                logger.error(f"Tool execution failed: {e}", exc_info=True)
        else:
            logger.error(f"Tool '{target_tool_name}' not found!")
            
    except Exception as e:
        logger.error("FAILURE! Exception occurred:", exc_info=True)
    finally:
        # Cleanup if possible (McpToolset doesn't have explicit close, but we can exit)
        pass

if __name__ == "__main__":
    asyncio.run(verify_connection())

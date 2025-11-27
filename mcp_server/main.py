from mcp.server.fastmcp import FastMCP
import logging

# Configure logging to file and console
import os
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'mcp_server.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        # logging.StreamHandler()  <-- REMOVED to prevent stdout/stderr corruption of MCP protocol
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"MCP Server logging to: {log_file}")

# Import mcp instance from tools (use absolute import to support both script and module execution)
try:
    from .tools import mcp
    from . import tools
except ImportError:
    # Running as a script, not as a module
    import tools
    from tools import mcp

if __name__ == "__main__":
    try:
        logger.info("Starting MCP Server...")
        logger.info("About to call mcp.run() with stdio transport")
        logger.info(f"MCP instance: {mcp}")
        logger.info(f"Registered tools: {len(mcp._tools) if hasattr(mcp, '_tools') else 'unknown'}")
        mcp.run()
        logger.info("mcp.run() completed (this should not be reached in stdio mode)")
    except Exception as e:
        logger.critical(f"MCP Server failed to start or crashed: {e}", exc_info=True)
        raise

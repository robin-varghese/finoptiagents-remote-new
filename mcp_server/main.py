from mcp.server.fastmcp import FastMCP
import logging

# Configure logging to file and console
import os
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'mcp_server.log')

# Configure base logging first
from app.utils.logging_config import setup_logging
setup_logging()

# Add FileHandler specifically for MCP server persistence
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

# IMPORTANT: Remove SteamHandler if present to avoid corrupting MCP stdio protocol? 
# setup_logging uses StreamHandler(sys.stdout). For MCP over stdio, this is FATAL.
# We must ensure root logger does NOT log to stdout for MCP server.
# Let's override the root logger handlers for this specific entry point.

root_logger = logging.getLogger()
# Remove all handlers that stream to stdout/stderr
for h in root_logger.handlers[:]:
    if isinstance(h, logging.StreamHandler):
        root_logger.removeHandler(h)

# Ensure only file handler remains
if file_handler not in root_logger.handlers:
    root_logger.addHandler(file_handler)
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

import os
import logging
from typing import List, Optional
from mcp_server.external.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# Configuration
DOCKER_IMAGE = "gcloud-mcp-image"
MOUNT_PATH = f"{os.path.expanduser('~')}/.config/gcloud:/root/.config/gcloud"

class GCloudMCPClient(MCPClient):
    """
    Specialized client for the GCloud MCP server.
    """
    def __init__(self):
        super().__init__(
            docker_image=DOCKER_IMAGE,
            mount_path=MOUNT_PATH
        )

# Global client instance (lazy initialization recommended in real app, but simple here)
# We will use a context manager in the tool function to ensure connection is managed.

async def run_gcloud_command(args: List[str]) -> str:
    """
    Executes a Google Cloud CLI (gcloud) command via the MCP server.

    Args:
        args: A list of arguments for the gcloud command. 
              Example: ['compute', 'instances', 'list', '--project', 'my-project']
              Do NOT include 'gcloud' as the first argument.

    Returns:
        The output of the gcloud command (stdout) or error message.
    """
    logger.info(f"Executing gcloud command: gcloud {' '.join(args)}")
    
    # Use the client as a context manager for each call to ensure clean connection/disconnection
    # In a high-throughput scenario, we might want to keep the connection open.
    async with GCloudMCPClient() as client:
        try:
            result = await client.call_tool("run_gcloud_command", arguments={"args": args})
            return result
        except Exception as e:
            logger.error(f"Failed to execute gcloud command: {e}")
            return f"Error executing command: {str(e)}"

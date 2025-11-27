import asyncio
import os
import json
import logging
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class MCPClient:
    """
    A client for connecting to an MCP server running in a Docker container via stdio.
    """

    def __init__(self, docker_image: str, mount_path: str, env: Optional[Dict[str, str]] = None):
        """
        Initialize the MCP Client.

        Args:
            docker_image: The name of the Docker image to run.
            mount_path: The host path to mount into the container (e.g., ~/.config/gcloud).
            env: Optional environment variables to pass to the container.
        """
        self.docker_image = docker_image
        self.mount_path = mount_path
        self.env = env
        self.session: Optional[ClientSession] = None
        self.exit_stack = None

    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

    async def connect(self):
        """Establishes the connection to the MCP server."""
        if self.session:
            return

        server_params = StdioServerParameters(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "--network", "host",
                "-v", self.mount_path,
                self.docker_image
            ],
            env=self.env
        )
        
        logger.info(f"Connecting to MCP server: {self.docker_image}")
        
        # We need to manage the context managers manually since we want the session to persist
        # across tool calls if needed, or we can just use this client as a context manager itself.
        # For simplicity in this implementation, we'll use the client as a context manager.
        
        self.exit_stack = AsyncExitStack()
        
        try:
            # Enter stdio_client context
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            
            # Enter ClientSession context
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            
            await self.session.initialize()
            logger.info(f"Connected to MCP server: {self.docker_image}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.docker_image}: {e}")
            await self.close()
            raise

    async def close(self):
        """Closes the connection."""
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None
            self.session = None
            logger.info(f"Disconnected from MCP server: {self.docker_image}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """
        Calls a tool on the MCP server.

        Args:
            tool_name: The name of the tool to call.
            arguments: The arguments to pass to the tool.

        Returns:
            The result of the tool call.
        """
        if not self.session:
            raise RuntimeError("Client is not connected. Use 'async with client:' or call connect().")

        logger.info(f"Calling tool '{tool_name}' on {self.docker_image}")
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments or {})
            
            # Process result content
            output = []
            for content in result.content:
                if content.type == "text":
                    output.append(content.text)
                else:
                    output.append(f"[{content.type} content]")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}")
            raise

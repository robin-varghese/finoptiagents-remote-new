#!/usr/bin/env python3
"""
Standalone MCP server test using the MCP Inspector.
This runs the MCP server and allows you to test it interactively.
"""
import subprocess
import sys
import os

# Get absolute paths
project_root = os.path.dirname(os.path.abspath(__file__))
mcp_server_script = os.path.join(project_root, "mcp_server", "main.py")

print("=" * 80)
print("MCP Server Standalone Test")
print("=" * 80)
print(f"Project root: {project_root}")
print(f"MCP server script: {mcp_server_script}")
print(f"Python: {sys.executable}")
print("=" * 80)

# Start MCP server
print("\nStarting MCP server...")
print("The server will run in stdio mode.")
print("You can test it with the MCP Inspector or by sending JSON-RPC messages.\n")

try:
    # Run the server directly
    subprocess.run(
        [sys.executable, mcp_server_script],
        cwd=project_root,
        env={
            **os.environ,
            "PYTHONPATH": project_root
        }
    )
except KeyboardInterrupt:
    print("\n\nMCP server stopped by user.")
except Exception as e:
    print(f"\n\nError running MCP server: {e}")
    import traceback
    traceback.print_exc()

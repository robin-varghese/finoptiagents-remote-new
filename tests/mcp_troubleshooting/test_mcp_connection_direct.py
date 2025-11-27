#!/usr/bin/env python3
"""
Direct test of the exact MCP configuration used in app/agent.py
to see what error occurs during MCP session creation.
"""
import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_mcp_connection():
    from google.adk.tools import McpToolset
    from google.adk.tools.mcp_tool import StdioServerParameters
    
    # Exact configuration from app/agent.py
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _current_dir = os.path.join(_current_dir, "app")
    _project_root = os.path.dirname(_current_dir)
    _mcp_server_script = os.path.join(_project_root, "mcp_server", "main.py")
    
    print("=" * 80)
    print("Testing MCP Connection with Exact app/agent.py Configuration")
    print("=" * 80)
    print(f"Python: {sys.executable}")
    print(f"MCP Script: {_mcp_server_script}")
    print(f"Project Root: {_project_root}")
    print("=" * 80)
    
    mcp_toolset = McpToolset(
        connection_params=StdioServerParameters(
            command=sys.executable,
            args=[_mcp_server_script],
            cwd=_project_root,
            env={
                **os.environ,
                "PYTHONPATH": _project_root
            }
        )
    )
    
    try:
        print("\nAttempting to get tools from MCP server...")
        tools = await mcp_toolset.get_tools()
        print(f"\n✅ SUCCESS! Got {len(tools)} tools from MCP server:")
        for tool in tools[:5]:  # Show first 5
            print(f"  - {tool.name}")
        if len(tools) > 5:
            print(f"  ... and {len(tools) - 5} more")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to get more details
        print("\n" + "=" * 80)
        print("Attempting to capture MCP server stderr...")
        print("=" * 80)
        
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, _mcp_server_script],
            cwd=_project_root,
            env={**os.environ, "PYTHONPATH": _project_root},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        import time
        time.sleep(2)
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=2)
        
        print("\n--- MCP Server STDOUT ---")
        print(stdout if stdout else "(empty)")
        print("\n--- MCP Server STDERR ---")
        print(stderr if stderr else "(empty)")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())

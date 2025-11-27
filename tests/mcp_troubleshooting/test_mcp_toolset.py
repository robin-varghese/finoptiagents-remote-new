#!/usr/bin/env python3
"""
Test MCP server connection using ADK's McpToolset.
This verifies the server can be connected to and lists all available tools.
"""
import asyncio
import sys
import os
from google.adk.tools import McpToolset
from mcp import StdioServerParameters

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
mcp_server_script = os.path.join(project_root, "mcp_server", "main.py")

async def test_mcp_server():
    print("=" * 80)
    print("Testing MCP Server Connection")
    print("=" * 80)
    print(f"MCP Server Script: {mcp_server_script}")
    print(f"Python: {sys.executable}")
    print("=" * 80)
    
    # Create MCP toolset
    mcp_toolset = McpToolset(
        connection_params=StdioServerParameters(
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
        print("\n1. Connecting to MCP server...")
        tools = await mcp_toolset.get_tools()
        
        print(f"\n✅ SUCCESS! Connected to MCP server")
        print(f"✅ Found {len(tools)} tools:")
        print()
        
        for i, tool in enumerate(tools, 1):
            print(f"  {i}. {tool.name}")
            if hasattr(tool, 'description') and tool.description:
                # Truncate long descriptions
                desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
                print(f"     {desc}")
        
        print("\n" + "=" * 80)
        print("✅ MCP Server Test PASSED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n" + "=" * 80)
        print("Detailed error:")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_mcp_server())

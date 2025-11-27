#!/usr/bin/env python3
"""
Test MCP server by actually calling a tool and getting a response.
This creates an McpToolset, connects to the server, and invokes a tool.
"""
import asyncio
import sys
import os
from google.adk.tools import McpToolset
from mcp import StdioServerParameters

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
mcp_server_script = os.path.join(project_root, "mcp_server", "main.py")

async def test_tool_invocation():
    print("=" * 80)
    print("Testing MCP Server Tool Invocation")
    print("=" * 80)
    print(f"MCP Server Script: {mcp_server_script}")
    print(f"Python: {sys.executable}")
    print("=" * 80)
    
    # Create MCP toolset
    print("\n1. Creating MCP toolset connection...")
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
        # Get tools
        print("2. Connecting to MCP server and fetching tools...")
        tools = await mcp_toolset.get_tools()
        print(f"   ✅ Connected! Found {len(tools)} tools\n")
        
        # Find the list_vm_instances tool
        list_vm_tool = None
        for tool in tools:
            if tool.name == "list_vm_instances":
                list_vm_tool = tool
                break
        
        if not list_vm_tool:
            print("❌ ERROR: Could not find 'list_vm_instances' tool")
            sys.exit(1)
        
        print(f"3. Found tool: {list_vm_tool.name}")
        print(f"   Description: {list_vm_tool.description[:100]}...")
        
        # Invoke the tool
        print("\n4. Invoking tool with test parameters...")
        print("   Parameters: {project_id='vector-search-poc', zone='us-central1-a'}")
        
        # Call the tool - McpTool requires arguments dict
        from google.adk.core import InvocationContext, Session
        from google.genai.types import Content, Part
        
        # Create a minimal context
        session = Session(app_name="test", user_id="test", session_id="test")
        context = InvocationContext(session=session, user_message=Content(parts=[Part(text="test")]))
        
        result = await list_vm_tool.run_async(
            parent_context=context,
            tool_input={
                "project_id": "vector-search-poc",
                "zone": "us-central1-a"
            }
        )
        
        print("\n5. Tool Response:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        print("\n✅ SUCCESS! Tool invocation completed")
        print("   The MCP server is working correctly end-to-end!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n" + "=" * 80)
        print("Detailed error:")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_tool_invocation())

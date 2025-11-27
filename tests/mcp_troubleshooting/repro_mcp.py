from mcp.types import Tool
try:
    t = Tool(name="test", inputSchema={})
    print(f"Tool created: {t}")
    print(f"Has outputSchema: {hasattr(t, 'outputSchema')}")
    print(f"outputSchema value: {t.outputSchema}")
except Exception as e:
    print(f"Error: {e}")

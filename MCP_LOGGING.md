# MCP Server Logging Guide

## 🎯 Current Architecture: Direct Imports for Streamlit

**Important Update:** The Streamlit UI (`app/playground.py`) now imports tools **directly** from `mcp_server/tools.py` instead of using the MCP subprocess. This guide is relevant for:
- Standalone MCP server usage (`verify_mcp_connection.py`)
- External MCP clients
- Future non-Streamlit deployments

For Streamlit, tools are imported as regular Python functions, so logging happens through the standard application logger.

-------

## 🚨 CRITICAL WARNING: NO CONSOLE LOGGING (MCP Server Mode)

**Only applicable when running MCP server as a subprocess.**

The MCP (Model Context Protocol) uses `stdio` (Standard Input/Output) to transport JSON-RPC messages between the client and server.

*   **Stdout is for Protocol Data ONLY.**
*   Any log message written to `stdout` will be interpreted as a protocol message.
*   Since log text is not valid JSON-RPC, this **corrupts the connection**, causing the client to ignore the data and eventually **timeout**.

## Logging Architecture

### For Streamlit UI (Direct Import Mode)
Tools imported directly in `app/agent.py` use Python's standard `logging` module:

```python
import logging
logger = logging.getLogger(__name__)

# In tools
logger.info("[TOOL CALL] list_vm_instances - Starting execution")
```

Logs appear in the Streamlit console output.

### For MCP Server (Subprocess Mode)
When running as an MCP server, all logs go to a file.

**Log Location:**
```
finoptiagents-remote-new/logs/mcp_server.log
```

**Configuration (`mcp_server/main.py`):**
```python
# ✅ CORRECT: File-only logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file)  # ✅ Safe - writes to file
        # logging.StreamHandler()      # ❌ BANNED - corrupts MCP stdio
    ]
)
```

## Viewing Logs

### Streamlit UI
Logs appear in the terminal where you ran `make playground`:
```bash
# Streamlit console shows:
2025-11-23 21:45:10 - app.agent - INFO - [TOOL CALL] run_bq_query...
```

### MCP Server (Standalone)
Tail the log file in a separate terminal:
```bash
tail -f logs/mcp_server.log
```

## Debugging Tips

### Streamlit UI Issues
1. Check the Streamlit terminal output
2. Verify imports in `app/agent.py` are correct
3. Look for Python exceptions in the UI error display

### MCP Server Issues (Standalone Mode)
1. **Check `logs/mcp_server.log`** for the full picture
2. If the log shows "Processing request..." but client times out:
   - Ensure no `print()` statements were added
   - Check for accidental `logging.StreamHandler()` additions
3. If no logs appear:
   - Server didn't start (check subprocess errors)
   - Log file permissions issue

### Adding New Tools
When adding tools to `mcp_server/tools.py`:

```python
import logging
logger = logging.getLogger(__name__)

@mcp.tool()
def my_new_tool(param: str) -> str:
    """Tool description."""
    logger.info(f"[TOOL CALL] my_new_tool - Starting")
    logger.info(f"[TOOL INPUT] param='{param}'")
    
    try:
        result = do_something(param)
        logger.info(f"[TOOL RESPONSE] my_new_tool - Success")
        return result
    except Exception as e:
        logger.error(f"[TOOL ERROR] my_new_tool - Failed: {e}")
        raise
```

**Never use:**
- ❌ `print(...)` anywhere in tool code
- ❌ `logging.StreamHandler()` when running as MCP server

## Architecture Summary

```
┌─────────────────────────┐
│   Streamlit UI          │
│   (app/playground.py)   │
└───────────┬─────────────┘
            │
            │ Direct Python import
            ▼
┌─────────────────────────┐
│   app/agent.py          │
│   ALL_TOOLS = [...]     │ ◄─── Direct function references
└───────────┬─────────────┘
            │
            │ from mcp_server.tools import ...
            ▼
┌─────────────────────────┐
│  mcp_server/tools.py    │  ◄─── 13 tool functions
│  - run_bq_query()       │      Standard Python logging
│  - send_email()         │
│  - list_vm_instances()  │
│  - ...                  │
└─────────────────────────┘


Standalone MCP Mode (verify_mcp_connection.py):

┌─────────────────────────┐
│  External Client        │
└───────────┬─────────────┘
            │
            │ stdio JSON-RPC
            ▼
┌─────────────────────────┐
│  mcp_server/main.py     │  ◄─── FastMCP subprocess
│  (MCP Server)           │      File-only logging
└───────────┬─────────────┘      (logs/mcp_server.log)
            │
            │ @mcp.tool() decorators
            ▼
┌─────────────────────────┐
│  mcp_server/tools.py    │
└─────────────────────────┘
```

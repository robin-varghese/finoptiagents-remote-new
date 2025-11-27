# MCP Troubleshooting Scripts

This directory contains scripts that were created during the development and troubleshooting of the MCP (Model Context Protocol) integration.

## Purpose

These scripts were used to:
- Debug MCP server connection issues
- Test MCP toolset functionality in isolation
- Verify tool execution and responses
- Troubleshoot Streamlit event loop conflicts

## Scripts

### Verification Scripts

- **`verify_mcp_connection.py`** - Primary verification script that tests MCP server connection, tool listing, and execution. Includes all necessary monkeypatches.
  ```bash
  conda run -n googleagentdevkit-new-nov-2025 python tests/mcp_troubleshooting/verify_mcp_connection.py
  ```

### Test Scripts

- **`test_mcp_agent.py`** - Tests the full agent with MCP tools
- **`test_mcp_toolset.py`** - Tests McpToolset creation and tool listing
- **`test_mcp_tool_call.py`** - Tests individual tool execution
- **`test_mcp_connection_direct.py`** - Direct MCP connection test
- **`test_mcp_exact_config.py`** - Tests with exact configuration matching production
- **`test_list_vms.py`** - Deprecated VM listing test

### Debug Scripts

- **`debug_mcp_server.py`** - Debug script for MCP server issues
- **`run_mcp_server.py`** - Standalone MCP server runner
- **`repro_mcp.py`** - Script to reproduce specific MCP issues

### Utilities

- **`mcp_wrapper.sh`** - Bash wrapper to capture MCP server stdout for debugging
- **`test_genai_module.py`** - Tests Google GenAI module imports
- **`test_genai_submodules.py`** - Tests GenAI submodule structure

## Historical Context

The MCP integration went through several iterations:

1. **Initial MCP Mode**: Attempted to use MCP server as subprocess from Streamlit
2. **Troubleshooting Phase**: Created these scripts to isolate issues
3. **Final Solution**: Switched to direct imports for Streamlit UI (see `app/agent.py`)

The MCP server (`mcp_server/main.py`) remains functional for standalone use, but Streamlit now uses direct Python imports to avoid event loop conflicts.

## Current Architecture

- **Streamlit UI**: Direct imports from `mcp_server/tools.py`
- **Standalone/External**: MCP server subprocess mode (verified with these scripts)

For production architecture details, see:
- `README.md` - MCP Architecture section
- `MCP_LOGGING.md` - Logging guide
- `mcp_server_deployment_summary.txt` - Complete documentation

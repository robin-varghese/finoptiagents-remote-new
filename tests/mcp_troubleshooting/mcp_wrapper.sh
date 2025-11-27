#!/bin/bash
# Wrapper script to launch MCP server and capture stdout for debugging
# This helps verify if the server is actually writing data to stdout

# Path to the python executable in the conda environment
PYTHON_EXEC="/Users/robinkv/miniconda3/envs/googleagentdevkit-new-nov-2025/bin/python3"

# Path to the MCP server script
SERVER_SCRIPT="/Users/robinkv/dev_workplace/all_codebase/finoptiagents_remote_new/finoptiagents-remote-new/mcp_server/main.py"

# Log file for captured stdout
LOG_FILE="/tmp/mcp_stdout.log"

# Launch the server with unbuffered output (-u) and tee stdout to the log file
# stderr is redirected to the same log file for completeness
exec $PYTHON_EXEC -u $SERVER_SCRIPT 2>&1 | tee -a $LOG_FILE

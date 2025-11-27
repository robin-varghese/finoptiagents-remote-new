#!/usr/bin/env python3
"""
Debug script to test MCP server startup and capture any errors.
"""
import subprocess
import sys
import os

# Set up the environment
env = {
    **os.environ,
    "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))
}

print("=" * 80)
print("Testing MCP Server Startup")
print("=" * 80)
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"PYTHONPATH: {env.get('PYTHONPATH')}")
print("=" * 80)

# Try to run the MCP server
try:
    print("\nAttempting to start MCP server subprocess...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.main"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a bit for startup
    import time
    time.sleep(2)
    
    # Check if it's still running
    poll_result = proc.poll()
    if poll_result is not None:
        print(f"\n❌ MCP server exited with code: {poll_result}")
        stdout, stderr = proc.communicate()
        print("\n--- STDOUT ---")
        print(stdout)
        print("\n--- STDERR ---")
        print(stderr)
    else:
        print("\n✅ MCP server appears to be running")
        proc.terminate()
        stdout, stderr = proc.communicate()
        if stderr:
            print("\n--- STDERR (on termination) ---")
            print(stderr)
        
except Exception as e:
    print(f"\n❌ Error starting MCP server: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

#!/usr/bin/env python3
"""
Test MCP server startup using the exact same parameters as app/agent.py
"""
import subprocess
import sys
import os
import time

# Mimic the exact setup from app/agent.py
_current_dir = os.path.dirname(os.path.abspath(__file__))
_current_dir = os.path.join(_current_dir, "app")  # Since this script is in project root
_project_root = os.path.dirname(_current_dir)
_mcp_server_script = os.path.join(_project_root, "mcp_server", "main.py")

env = {
    **os.environ,
    "PYTHONPATH": _project_root
}

print("=" * 80)
print("Testing MCP Server with app/agent.py Configuration")
print("=" * 80)
print(f"Python executable: {sys.executable}")
print(f"MCP server script: {_mcp_server_script}")
print(f"Working directory (cwd): {_project_root}")
print(f"PYTHONPATH: {env.get('PYTHONPATH')}")
print("=" * 80)

# Verify the script exists
if not os.path.exists(_mcp_server_script):
    print(f"\n❌ MCP server script not found at: {_mcp_server_script}")
    sys.exit(1)

print(f"\n✅ MCP server script exists")

# Try to run it
try:
    print("\nLaunching MCP server...")
    proc = subprocess.Popen(
        [sys.executable, _mcp_server_script],
        cwd=_project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a bit
    time.sleep(3)
    
    # Check status
    poll_result = proc.poll()
    if poll_result is not None:
        print(f"\n❌ MCP server exited with code: {poll_result}")
        stdout, stderr = proc.communicate(timeout=2)
        print("\n--- STDOUT ---")
        print(stdout if stdout else "(empty)")
        print("\n--- STDERR ---")
        print(stderr if stderr else "(empty)")
        sys.exit(1)
    else:
        print("\n✅ MCP server is running!")
        print("   Terminating for cleanup...")
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=2)
            if stderr and stderr.strip():
                print("\n--- STDERR (during run) ---")
                print(stderr)
        except:
            proc.kill()
            
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" *  80)
print("✅ Test completed successfully!")
print("=" * 80)

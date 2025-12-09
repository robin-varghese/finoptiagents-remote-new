#!/usr/bin/env python3
"""
Test Script for GCloud Recommender Commands via MCP Server

This script tests whether the gcloud MCP server can execute various
gcloud recommender commands for cost optimization.
"""

import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuration
DOCKER_IMAGE = "gcloud-mcp-image"
MOUNT_PATH = f"{os.path.expanduser('~')}/.config/gcloud:/root/.config/gcloud"

# Server parameters
server_params = StdioServerParameters(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "--network", "host",
        "-v", MOUNT_PATH,
        DOCKER_IMAGE
    ],
    env=None
)

async def get_project_id():
    """Get project ID from gcloud config."""
    import subprocess
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"⚠️  Could not get project from gcloud config: {e}")
    return None

async def test_recommender_commands():
    """Test various gcloud recommender commands."""
    
    print("=" * 70)
    print("GCloud Recommender Commands Test Suite")
    print("=" * 70)
    
    # Get project ID from environment or gcloud config
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        print("\n⏳ No GCP_PROJECT_ID environment variable, checking gcloud config...")
        project_id = await get_project_id()
    
    if not project_id:
        print("❌ Could not determine project ID. Please set GCP_PROJECT_ID or run 'gcloud config set project PROJECT_ID'")
        return
    
    print(f"\n📋 Testing with Project ID: {project_id}\n")
    
    # Define test cases
    test_cases = [
        {
            "name": "List Idle VM Recommender",
            "description": "Check for idle/unused VM instances",
            "args": [
                "recommender", "recommendations", "list",
                f"--project={project_id}",
                "--location=global",
                "--recommender=google.compute.instance.IdleResourceRecommender",
                "--format=json"
            ]
        },
        {
            "name": "List VM Machine Type Recommender (us-central1-a)",
            "description": "Check for VM machine type rightsizing opportunities",
            "args": [
                "recommender", "recommendations", "list",
                f"--project={project_id}",
                "--location=us-central1-a",
                "--recommender=google.compute.instance.MachineTypeRecommender",
                "--format=json"
            ]
        },
        {
            "name": "List Idle Cloud SQL Recommender (us-central1)",
            "description": "Check for idle/unused Cloud SQL instances",
            "args": [
                "recommender", "recommendations", "list",
                f"--project={project_id}",
                "--location=us-central1",
                "--recommender=google.cloudsql.instance.IdleRecommender",
                "--format=json"
            ]
        },
        {
            "name": "List Idle Persistent Disk Recommender",
            "description": "Check for idle/unused persistent disks",
            "args": [
                "recommender", "recommendations", "list",
                f"--project={project_id}",
                "--location=global",
                "--recommender=google.compute.disk.IdleResourceRecommender",
                "--format=json"
            ]
        },
        {
            "name": "List All Recommenders",
            "description": "List all available recommenders for the project",
            "args": [
                "recommender", "recommenders", "list",
                f"--project={project_id}",
                "--format=table(name,displayName)"
            ]
        }
    ]
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize connection
                await session.initialize()
                print("✅ Connected to GCloud MCP Server\n")
                
                # List available tools
                tools = await session.list_tools()
                print(f"Available Tools: {[t.name for t in tools.tools]}\n")
                print("=" * 70)
                
                # Run each test case
                results = []
                
                for i, test in enumerate(test_cases, 1):
                    print(f"\n[Test {i}/{len(test_cases)}] {test['name']}")
                    print(f"Description: {test['description']}")
                    print(f"Command: gcloud {' '.join(test['args'])}")
                    print("-" * 70)
                    
                    try:
                        # Execute the command
                        result = await session.call_tool(
                            "run_gcloud_command",
                            arguments={"args": test['args']}
                        )
                        
                        # Analyze result
                        success = True
                        output_text = ""
                        
                        for content in result.content:
                            if content.type == "text":
                                output_text = content.text
                                # Check for common error patterns
                                if "ERROR:" in output_text or "PERMISSION_DENIED" in output_text:
                                    success = False
                        
                        # Store result
                        results.append({
                            "test": test['name'],
                            "success": success,
                            "output": output_text[:500]  # Limit output length
                        })
                        
                        # Display result
                        if success:
                            print("✅ Command executed successfully")
                            if output_text:
                                # Try to parse as JSON and count recommendations
                                import json
                                try:
                                    data = json.loads(output_text)
                                    if isinstance(data, list):
                                        print(f"📊 Found {len(data)} recommendation(s)")
                                        if len(data) > 0:
                                            print(f"   Sample: {data[0].get('name', 'N/A')}")
                                    elif isinstance(data, dict):
                                        print(f"📊 Response: {list(data.keys())}")
                                except json.JSONDecodeError:
                                    # Not JSON, show first few lines
                                    lines = output_text.split('\n')[:3]
                                    for line in lines:
                                        if line.strip():
                                            print(f"   {line[:80]}")
                        else:
                            print("❌ Command failed")
                            # Show error details
                            error_lines = output_text.split('\n')[:5]
                            for line in error_lines:
                                if line.strip():
                                    print(f"   {line}")
                    
                    except Exception as e:
                        print(f"❌ Exception occurred: {str(e)}")
                        results.append({
                            "test": test['name'],
                            "success": False,
                            "output": str(e)
                        })
                
                # Summary
                print("\n" + "=" * 70)
                print("TEST SUMMARY")
                print("=" * 70)
                
                successful = sum(1 for r in results if r['success'])
                total = len(results)
                
                print(f"\nTotal Tests: {total}")
                print(f"Passed: {successful}")
                print(f"Failed: {total - successful}")
                print(f"Success Rate: {(successful/total*100):.1f}%")
                
                print("\nDetailed Results:")
                for r in results:
                    status = "✅ PASS" if r['success'] else "❌ FAIL"
                    print(f"  {status} - {r['test']}")
                
                print("\n" + "=" * 70)
                print("CONCLUSION")
                print("=" * 70)
                
                if successful == total:
                    print("✅ All recommender commands are supported!")
                    print("   The gcloud MCP server can execute gcloud recommender commands.")
                elif successful > 0:
                    print("⚠️  Some recommender commands work, but some failed.")
                    print("   Check the errors above for details (may be permissions or missing resources).")
                else:
                    print("❌ All recommender commands failed.")
                    print("   There may be a configuration or permission issue.")
                
                print("\n💡 Next Steps:")
                print("   1. If commands failed due to missing resources, that's expected in a test project")
                print("   2. If permission errors occurred, check IAM roles (roles/recommender.viewer)")
                print("   3. The MCP server itself CAN execute these commands - the question is permissions")
                
    except Exception as e:
        print(f"\n❌ Failed to connect to MCP server: {e}")
        print("   Ensure the Docker image 'gcloud-mcp-image' is built and running")
        print("   Build command: docker build -t gcloud-mcp-image .")

if __name__ == "__main__":
    try:
        asyncio.run(test_recommender_commands())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)

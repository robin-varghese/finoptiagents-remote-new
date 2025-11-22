import fetch from 'node-fetch';

// Use localhost when running with gcloud run services proxy
const SERVICE_URL = 'http://localhost:8080'; 

async function testMcpServer() {
  try {
    const response = await fetch(`${SERVICE_URL}/mcp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        command: 'run_gcloud_command',
        args: ['config', 'list'],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
    }

    const data = await response.json();
    console.log('MCP Server Response:', data);
  } catch (error) {
    console.error('Error testing MCP server:', error);
  }
}

testMcpServer();

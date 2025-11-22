const express = require('express');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = process.env.PORT || 8080;
const host = '0.0.0.0';

app.use(express.json()); // Enable JSON body parsing

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

// MCP endpoint
app.post('/mcp', (req, res) => {
  const mcpArgs = req.body.args || []; // Assuming args are passed in the request body
  const command = req.body.command; // Assuming command is passed in the request body

  if (!command) {
    return res.status(400).send('Missing command in request body');
  }

  // Construct the full command to execute gcloud-mcp
  // We'll use npx to run the gcloud-mcp package
  const npxArgs = ['-y', '@google-cloud/gcloud-mcp', command, ...mcpArgs];

  const mcpProcess = spawn('npx', npxArgs);

  let stdout = '';
  let stderr = '';

  mcpProcess.stdout.on('data', (data) => {
    stdout += data.toString();
  });

  mcpProcess.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  mcpProcess.on('close', (code) => {
    if (code === 0) {
      res.status(200).json({ stdout, stderr });
    } else {
      res.status(500).json({ error: `MCP command failed with code ${code}`, stdout, stderr });
    }
  });

  mcpProcess.on('error', (err) => {
    console.error('Failed to start MCP process:', err);
    res.status(500).json({ error: 'Failed to start MCP process', details: err.message });
  });
});

app.listen(port, host, () => {
  console.log(`MCP wrapper server listening on ${host}:${port}`);
});

# finoptiagents-remote-new

A base ReAct agent built with Google's Agent Development Kit (ADK)
Agent generated with [`googleCloudPlatform/agent-starter-pack`](https://github.com/GoogleCloudPlatform/agent-starter-pack) version `0.11.2`

## Project Structure

This project is organized as follows:

```
finoptiagents-remote-new/
├── app/                 # Core application code
│   ├── agent.py         # Main agent logic with direct tool imports
│   ├── playground.py    # Streamlit UI for local testing
│   ├── agent_engine_app.py # Agent Engine application logic
│   └── utils/           # Utility functions and helpers
├── mcp_server/          # MCP Server and tool implementations
│   ├── main.py          # FastMCP server (for standalone/external use)
│   ├── tools.py         # All 13 tool implementations
│   ├── config.py        # Configuration and Secret Manager integration
│   └── rag_utils.py     # RAG utility functions
├── logs/                # Application logs
│   └── mcp_server.log   # MCP server logs (when running in MCP mode)
├── .cloudbuild/         # CI/CD pipeline configurations for Google Cloud Build
├── deployment/          # Infrastructure and deployment scripts
├── notebooks/           # Jupyter notebooks for prototyping and evaluation
├── tests/               # Unit, integration, and load tests
│   └── mcp_troubleshooting/  # MCP debugging and verification scripts
├── verify_mcp_connection.py  # Script to test MCP server mode (moved to tests/mcp_troubleshooting/)
├── Makefile             # Makefile for common commands
├── GEMINI.md            # AI-assisted development guide
├── MCP_LOGGING.md       # MCP logging guide and best practices
├── mcp_server_deployment_summary.txt  # MCP architecture documentation
└── pyproject.toml       # Project dependencies and configuration
```

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager - [Install](https://docs.astral.sh/uv/getting-started/installation/)
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)
- **Terraform**: For infrastructure deployment - [Install](https://developer.hashicorp.com/terraform/downloads)
- **make**: Build automation tool - [Install](https://www.gnu.org/software/make/) (pre-installed on most Unix-based systems)


## Quick Start (Local Testing)

Install required packages and launch the local development environment:

```bash
make install && make playground
```

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `make install`       | Install all required dependencies using uv                                                  |
| `make playground`    | Launch Streamlit interface for testing agent locally and remotely |
| `make backend`       | Deploy agent to Agent Engine |
| `make test`          | Run unit and integration tests                                                              |
| `make lint`          | Run code quality checks (codespell, ruff, mypy)                                             |
| `make setup-dev-env` | Set up development environment resources using Terraform                         |
| `uv run jupyter lab` | Launch Jupyter notebook                                                                     |

For full command options and usage, refer to the [Makefile](Makefile).


## Usage

This template follows a "bring your own agent" approach - you focus on your business logic, and the template handles everything else (UI, infrastructure, deployment, monitoring).

1. **Prototype:** Build your Generative AI Agent using the intro notebooks in `notebooks/` for guidance. Use Vertex AI Evaluation to assess performance.
2. **Integrate:** Import your agent into the app by editing `app/agent.py`.
3. **Test:** Explore your agent functionality using the Streamlit playground with `make playground`. The playground offers features like chat history, user feedback, and various input types, and automatically reloads your agent on code changes.
4. **Deploy:** Set up and initiate the CI/CD pipelines, customizing tests as necessary. Refer to the [deployment section](#deployment) for comprehensive instructions. For streamlined infrastructure deployment, simply run `uvx agent-starter-pack setup-cicd`. Check out the [`agent-starter-pack setup-cicd` CLI command](https://googlecloudplatform.github.io/agent-starter-pack/cli/setup_cicd.html). Currently supports GitHub with both Google Cloud Build and GitHub Actions as CI/CD runners.
5. **Monitor:** Track performance and gather insights using Cloud Logging, Tracing, and the Looker Studio dashboard to iterate on your application.

The agent is designed to assist with a wide range of Google Cloud financial operations and resources management tasks, now powered by an advanced FinOps suite:

### 1. Cost Optimization & Recommenders (NEW)
Comprehensive scanning using 16+ Google Cloud Recommenders across Global, Regional, and Zonal scopes:
*   **Compute Engine**:
    *   **Idle VM Recommender**: Identification and termination of zombie instances.
    *   **Rightsizing Recommender**: Recommendations for over/under-provisioned VMs and Instance Groups (MIGs).
    *   **Idle Resource Recommender**: Detection of unused IP addresses, Disks, and Custom Images.
*   **Google Kubernetes Engine (GKE)**:
    *   **Cluster Diagnosis**: Identification of idle, over-provisioned, or under-provisioned clusters and node pools.
*   **Cloud SQL**:
    *   **Instance Rightsizing**: Detection of idle, over-provisioned, and under-provisioned database instances.
*   **Cloud Run**:
    *   **Service Optimization**: Cost and CPU allocation recommendations for serverless workloads.
*   **Committed Use Discounts (CUDs)**:
    *   **Commitment Analysis**: Insights into Spend-based and Usage-based commitment utilization.

### 2. FinOps Data Analysis & Reporting
*   **BigQuery Integration**: The agent executes read-only SQL queries against your BigQuery datasets to answer complex questions about:
    *   **Cloud Costs**: Daily/Monthly spend tracking by project, service, or label.
    *   **Resource Usage**: CPU, Memory, and Network utilization trends.
    *   **Compliance**: Verification of resource adherence to governance policies.

### 3. Automated Auditing & Logging (NEW)
*   **BigQuery Auditing Plugin**: A custom-built plugin automatically logs every agent action, tool call, and state change to `agent_analytics_log`, creating an immutable audit trail.
*   **Cost Savings Ledger**:
    *   **Dedicated Savings Log**: Every successful cost-saving operation (e.g., deleting a VM, rightsizing) is automatically logged to the `cost_savings_log` table.
    *   **Impact Tracking**: Tracks realized savings in USD, linked to specific operation IDs for ROI analysis.
*   **VM Deletion Auditing**: Specialized auditing for "Who deleted what and when?" queries, parsing complex JSON logs from `vm_deletion_log`.

### 4. Visualization & Design Review
*   **Data Visualization**: Automatically generate interactive bar, line, and pie charts from BigQuery data.
*   **Implementation Review (RAG)**: Compare deployed cloud resources against design documents using retrieval-augmented generation.

### Development Workflow

1.  **Customize**: Modify the agent's tools, prompts, and orchestration logic in `app/agent.py` to fit your specific business needs.
2.  **Test**: Explore your agent's functionality using the Streamlit playground with `make playground`. The playground offers features like chat history, user feedback, and various input types, and automatically reloads your agent on code changes.
3.  **Deploy:** Set up and initiate the CI/CD pipelines, customizing tests as necessary. Refer to the deployment section for comprehensive instructions. For streamlined infrastructure deployment, simply run `uvx agent-starter-pack setup-cicd`. Check out the `agent-starter-pack setup-cicd` CLI command. Currently supports GitHub with both Google Cloud Build and GitHub Actions as CI/CD runners.
4.  **Monitor:** Track performance and gather insights using Cloud Logging, Tracing, and the Looker Studio dashboard to iterate on your application.

The project includes a `GEMINI.md` file that provides context for AI tools like Gemini CLI when asking questions about your template.

## MCP Architecture (Model Context Protocol)

This project uses a **hybrid tool architecture** combining direct imports and MCP server capabilities.

### Tool Organization

All 15 tools are defined in `mcp_server/tools.py`:
- `scan_cost_recommendations` - Unified cost scanner (16+ recommenders)
- `log_savings_impact` - Logs realized savings to BigQuery
- `run_bq_query` - Execute BigQuery queries
- `send_email` - Send emails via Cloud Function
- `list_vm_instances` - List GCP VMs
- `delete_vm_instance` - Delete GCP VMs
- `generate_chart_from_data` - Create visualizations
- `call_cpu_utilization_agent` - Call remote Vertex AI agent
- RAG tools: `rag_query`, `create_corpus`, `add_data`, `delete_corpus`, `delete_document`, `get_corpus_info`, `list_corpora`

### Dual-Mode Architecture

#### Streamlit UI (Production)
**Mode:** Direct Python imports  
**Implementation:** Tools imported directly in `app/agent.py`

```python
from mcp_server.tools import run_bq_query, send_email, list_vm_instances, ...
ALL_TOOLS = [run_bq_query, send_email, ...]  # Used by agents
```

**Why?** Streamlit's event loop conflicts with MCP subprocess stdio communication. Direct imports provide immediate, reliable access to tools.

#### Standalone/External Clients
**Mode:** MCP Server subprocess  
**Implementation:** FastMCP server in `mcp_server/main.py`

```bash
# Test MCP server mode
conda run -n googleagentdevkit-new-nov-2025 python tests/mcp_troubleshooting/verify_mcp_connection.py
```

**Use Cases:**
- Testing tool functionality in isolation
- External MCP clients
- Future non-Streamlit deployments

### Key Files

- **`mcp_server/tools.py`** - All tool implementations (13 tools)
- **`mcp_server/main.py`** - FastMCP server (for standalone use)
- **`mcp_server/config.py`** - Configuration and Secret Manager integration
- **`app/agent.py`** - Agent definitions with direct tool imports
- **`tests/mcp_troubleshooting/`** - MCP debugging and verification scripts
- **`MCP_LOGGING.md`** - Logging guide for both modes
- **`mcp_server_deployment_summary.txt`** - Complete architecture documentation

### Adding New Tools

1. Define the tool in `mcp_server/tools.py`:
```python
@mcp.tool()
def my_new_tool(param: str) -> str:
    """Tool description for LLM."""
    logger.info(f"[TOOL CALL] my_new_tool - Starting")
    # Implementation
    return result
```

2. Import in `app/agent.py`:
```python
from mcp_server.tools import my_new_tool
ALL_TOOLS = [..., my_new_tool]  # Add to list
```

3. Tool is now available in both Streamlit UI and MCP server mode.

### Logging

- **Streamlit UI:** Logs appear in terminal where `make playground` runs
- **MCP Server:** File-based logs in `logs/mcp_server.log` (view with `tail -f logs/mcp_server.log`)

**Important:** Never use `print()` in tool code or add `logging.StreamHandler()` when running MCP server (corrupts stdio protocol).

For complete details, see `MCP_LOGGING.md` and `mcp_server_deployment_summary.txt`.


## Deployment

> **Note:** For a streamlined one-command deployment of the entire CI/CD pipeline and infrastructure using Terraform, you can use the [`agent-starter-pack setup-cicd` CLI command](https://googlecloudplatform.github.io/agent-starter-pack/cli/setup_cicd.html). Currently supports GitHub with both Google Cloud Build and GitHub Actions as CI/CD runners.

### Dev Environment

You can test deployment towards a Dev Environment using the following command:

```bash
gcloud config set project <your-dev-project-id>
make backend
```


The repository includes a Terraform configuration for the setup of the Dev Google Cloud project.
See [deployment/README.md](deployment/README.md) for instructions.

### Production Deployment

The repository includes a Terraform configuration for the setup of a production Google Cloud project. Refer to [deployment/README.md](deployment/README.md) for detailed instructions on how to deploy the infrastructure and application.


## Monitoring and Observability
> You can use [this Looker Studio dashboard](https://lookerstudio.google.com/reporting/46b35167-b38b-4e44-bd37-701ef4307418/page/tEnnC
) template for visualizing events being logged in BigQuery. See the "Setup Instructions" tab to getting started.

The application uses OpenTelemetry for comprehensive observability with all events being sent to Google Cloud Trace and Logging for monitoring and to BigQuery for long term storage.

Access the Application Correctly (Browser Change)
This is the most important step.
Stop your Python server.
Restart it: uvicorn main:app --host 0.0.0.0 --port 8000
In your Chrome browser, go to this specific URL:
http://localhost:8000
or
http://127.0.0.1:8000
Do NOT use http://0.0.0.0:8000 in the browser.
Once the page loads, perform a hard refresh one last time to be absolutely sure you have the latest code:
Windows/Linux: Ctrl + Shift + R
Mac: Cmd + Shift + R
By accessing the app via localhost, you provide the secure context needed for the AudioWorklet API to function. Combined with the code changes that prevent crashes, both the audio errors and the WebSocket connection problems should now be resolved.

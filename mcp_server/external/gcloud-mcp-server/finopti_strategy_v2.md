# Strategy: FinOptiAgents Unified Integration (v2)

This updated strategy outlines the architectural approach to integrate exhaustive cost optimization features into **FinOptiAgents**, with a specific focus on accurate savings calculation and MCP-exclusive execution.

## 1. Architectural Goals
- **Unified Interface**: A single entry point for all cost optimization domains.
- **MCP-Exclusive**: All interactions with Google Cloud must be executed via the **GCloud MCP Server**.
- **Accurate Financials**: Savings reporting must align exactly with the Google Cloud Console.

## 2. Savings Calculation Strategy

Accurate reporting of "money saved" is critical. We will **not** attempt manual price calculations (complex SKUs). Instead, we strictly rely on the `primaryImpact` data provided directly by the Recommender API.

### 2.1 Data Source
Every recommendation JSON payload contains a standardized cost projection:
```json
"primaryImpact": {
  "category": "COST",
  "costProjection": {
    "cost": {
      "currencyCode": "USD",
      "units": "-15",   // Dollars (negative implies savings)
      "nanos": -500000000 // Cents
    },
    "duration": "2592000s" // Monthly (30 days)
  }
}
```

### 2.2 Calculation Logic (Pass-through & Aggregate)
1.  **Extraction**: The module extracts `units` and `nanos` from the recommendation JSON.
2.  **Normalization**: Convert to float: `abs(units + nanos/1e9)`.
3.  **Aggregation**: Sum these values to generate **"Total Monthly Potential Savings"**.

### 2.3 Metrics to Track
-   **Potential Savings**: Sum of all open recommendations found during the scan.
-   **Realized Savings**: Sum of the cost impact of *successfully executed* remediation actions.

## 3. System Design

### 3.1 Core Components (MCP-Driven)

1.  **`MCPConnector` (Transport)**:
    -   Exclusive gateway to the GCloud MCP Server.
    -   Proxies all `run_gcloud_command` calls.

2.  **`RecommenderWrappers` (Data + Cost)**:
    -   Fetches recommendations via MCP.
    -   **New Responsibility**: Parses `primaryImpact` to return a `SavingsEstimate` object with every recommendation.

3.  **`OptimizationModules`**:
    -   **Compute**, **Database**, **Container**, **Storage**.
    -   Uses `RecommenderWrappers` to get actionable items + cost data.

4.  **`FinOptiOrchestrator`**:
    -   Aggregates total savings across all modules.
    -   Manages the "Scan -> Plan -> Execute" workflow.

## 4. Implementation Strategy

### Phase 1: The Unified Scanner (MCP-Based)
**Goal**: Build capabilities to scan and calculate potential savings.
- Create `scanner.py` using MCP Stdio.
- Implement logic to sum up `primaryImpact` from all 4 domains.
- **Key Deliverable**: A tool that returns a report: *"Found 12 optimizations with $450/month potential savings"*.

### Phase 2: The Remediation Primitives (MCP-Based)
**Goal**: Implement execution logic via MCP commands.
- Verify MCP tool calls for `stop_instance`, `patch_instance`, etc.

### Phase 3: The Safety & Policy Layer
**Goal**: Prevent disasters.
- **Pre-flight Checks**: Verify state via MCP before acting.
- **Snapshotting**: Force snapshot via MCP before deletion.

## 5. Deployment Plan (Completed)
- **Mock Testing**: Test cost aggregation logic with sample JSONs.
- **Dev Verification**: Run distinct scans against `vector-search-poc`.

## 6. Execution Protocol (Implemented)

The system now operates under a strict **Scan -> Plan -> Log -> Execute** protocol:

1.  **Scan**: `gcloud_recommender_agent` uses `scan_cost_recommendations` to identify inefficiencies.
2.  **Plan**: Findings are presented to the user with calculated potential savings (USD).
3.  **Execute**: User authorizes remediation (e.g., "Delete that idle VM").
4.  **Log**:
    -   `compliance_logger_agent` automatically intercepts the action.
    -   Logs the operation to `cloud_operations_log` with `operation_source="admin_manual"` (if user triggered) or `"recommender_api"`.
    -   Logs the financial impact to `cost_savings_log` using the `primaryImpact` data.


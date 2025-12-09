# Data Schemas and Validation

This document outlines the data schemas for the BigQuery tables used by the FinOpti agents, the process for SQL query validation, and the Pydantic schemas used for structured agent outputs.

## 1. BigQuery Table Schemas

This section documents the schemas for the core BigQuery tables used in FinOps analysis.

### 1.1 `release_train_ticket`

Stores approved budgets and release plans for projects.

| Column Name | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `project_name` | STRING | The unique name of the project. | "project-alpha" |
| `budgeted_cost` | FLOAT64 | The total approved budget for the project. | 50000.00 |
| `planned_release_date` | STRING | The target quarter for the release. | "2025-Q1" |

### 1.2 `finops_cost_usage`

Contains actual cost and usage data for all resources.

| Column Name | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `project_name` | STRING | The name of the project the resource belongs to. | "project-alpha" |
| `month` | STRING | The month of the cost record (e.g., "YYYY-MM"). | "2025-01" |
| `total_cost` | FLOAT64 | The total cost incurred for the resource in that month. | 45000.00 |
| `resource_type`| STRING | The type of the resource. | "compute" |
| `utilization_pct`| FLOAT64 | The utilization percentage of the resource. | 0.65 |

### 1.3 `earb_review`

Logs the Enterprise Architecture Review Board (EARB) status for projects.

| Column Name | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `project_name` | STRING | The name of the project. | "project-alpha" |
| `review_status`| STRING | The current status of the EARB review. | "approved" |
| `approval_date`| STRING | The date of the EARB approval (e.g., "YYYY-MM-DD"). | "2024-12-01" |

### 1.4 `vm_deletion_log`

Provides an audit trail for VM deletion events.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `log_data` | JSON | A JSON object containing detailed information about the deletion event. See detailed schema below. |
| `embedding` | VECTOR | Vector embedding of the deletion event for semantic search (optional, not used in current queries). |

**JSON Structure of `log_data`:**

> [!IMPORTANT]
> The `log_data` column contains a **double-encoded JSON string**. When querying, you must use the double-parsing pattern:
> ```sql
> JSON_EXTRACT_SCALAR(PARSE_JSON(JSON_EXTRACT_SCALAR(log_data, '$')), '$.field_name')
> ```

**Fields within `log_data` JSON:**
- `vm_name` (STRING): Name of the deleted VM
- `user_id` (STRING): Email/username of the person who deleted the VM
- `deletion_timestamp_utc` (STRING): ISO 8601 timestamp (e.g., "2025-12-02T14:30:00.000Z")
- `zone` (STRING): GCP zone (e.g., "us-central1-a")
- `project_id` (STRING): GCP project ID

**Example `log_data` content:**
```json
{"vm_name": "test-instance-1", "user_id": "robin@example.com", "deletion_timestamp_utc": "2025-12-02T14:30:00.000Z", "zone": "us-central1-a", "project_id": "vector-search-poc"}
```


---

## 2. SQL Query Validation Process

To ensure accuracy and performance, all SQL queries used by specialist agents must go through a rigorous validation process:

1.  **Centralize Queries**: All specialist agent SQL queries are to be written and stored in a central `queries/` directory.
2.  **Validate Against BigQuery**: Each query must be validated against the actual BigQuery tables to ensure there are no syntax errors.
3.  **Test with Real Data**: Queries must be tested with production-like data to confirm they return the expected results.
4.  **Measure Performance**: All queries must execute in **under 3 seconds** to meet performance requirements.

---

## 3. Pydantic Output Schemas

To ensure reliable communication between agents, all specialist agents and managers use structured output schemas defined with Pydantic.

### 3.1 `BudgetVarianceResult`

Output schema for the `BudgetVarianceAgent`.

```python
from pydantic import BaseModel
from typing import Literal

class BudgetVarianceResult(BaseModel):
    projects_at_risk: list[dict]  # [{project_name, variance_pct, budgeted, actual}]
    variance_threshold: float = 0.10
    total_projects_analyzed: int
    severity: Literal["low", "medium", "high"]
```

### 3.2 `ComplianceResult`

Output schema for the `ComplianceAuditorAgent`.

```python
from pydantic import BaseModel
from typing import Literal

class ComplianceResult(BaseModel):
    non_compliant_projects: list[str]
    violation_type: Literal["missing_from_release_train", "missing_from_earb", "both"]
    recommended_actions: list[str]
    severity: Literal["low", "medium", "high", "critical"]
```

### 3.3 `UtilizationResult`

Output schema for the `UtilizationAnalystAgent`.

```python
from pydantic import BaseModel

class UtilizationResult(BaseModel):
    anomalous_environments: list[dict]  # [{project, prod_cost, lower_env_cost}]
    low_utilization_resources: list[dict]  # [{resource_id, utilization_pct, cost}]
    total_potential_savings: float
```

### 3.4 `OptimizationResult`

Output schema for the `OptimizationScoutAgent`.

```python
from pydantic import BaseModel

class OptimizationResult(BaseModel):
    top_cost_contributors: list[dict]  # [{resource_type, cost, utilization, savings_potential}]
    optimization_candidates: list[str]
    estimated_monthly_savings: float
```

### 3.5 `ReadinessResult`

Output schema for the `EnvironmentReadinessAgent`.

```python
from pydantic import BaseModel

class ReadinessResult(BaseModel):
    zombie_environments: list[dict]  # [{env_name, cost, days_idle, active_tickets}]
    justified_environments: list[str]
    total_zombie_cost: float
```

### 3.6 `FinOpsHealthReport`

Output schema for the `FinOpsAnalyticsManager`.

```python
from pydantic import BaseModel

class FinOpsHealthReport(BaseModel):
    budget_summary: BudgetVarianceResult
    compliance_summary: ComplianceResult
    utilization_summary: UtilizationResult
    optimization_summary: OptimizationResult
    readiness_summary: ReadinessResult
    overall_health_score: float  # 0-100
    critical_actions_required: list[str]
```

### 3.7 `AgentErrorResponse`

Standardized error response schema for all agents.

```python
from pydantic import BaseModel
from typing import Literal, Optional

class AgentErrorResponse(BaseModel):
    error_type: Literal["data_unavailable", "query_failed", "api_timeout"]
    error_message: str
    fallback_executed: bool
    partial_results: Optional[dict] = None
```

### 3.8 `VmDeletionAuditResult`

Output schema for the `VmDeletionAuditorAgent`.

```python
from pydantic import BaseModel
from typing import Optional

class VmDeletionAuditResult(BaseModel):
    deleted_vms: list[dict]  # [{vm_name, deleted_by, deletion_timestamp, zone, project_id}]
    total_deletions: int
    query_timeframe: str  # e.g., "today", "yesterday", "last 7 days"
    queried_user: Optional[str] = None  # If filtering by user
```

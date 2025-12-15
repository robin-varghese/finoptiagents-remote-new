# Strategy: FinOptiAgents Unified Integration

This strategy outlines the architectural approach to integrate exhaustive cost optimization features (Compute, SQL, GKE, Storage) into **FinOptiAgents** in a single consolidated release.

## 1. Architectural Goals
- **Unified Interface**: A single entry point for all cost optimization domains.
- **Extensibility**: Easy to add new recommenders in the future without refactoring core logic.
- **Safety**: "Scan first, act later" model with built-in safeguards against accidental data loss.

## 2. System Design

### 2.1 Core Components

We will structure the solution around a `RecommendationEngine` pattern:

1.  **`RecommenderClient` (Data Layer)**:
    -   Wraps the `gcloud recommender` API.
    -   Handles authentication and region iteration (Global vs. Regional checks).
    -   Normalizes output from different recommenders (e.g., standardizing VM vs. SQL JSON payloads).

2.  **`OptimizationModules` (Logic Layer)**:
    -   **`ComputeOptimizer`**: Handles Idle VMs, Rightsizing, IPs, Disks.
    -   **`DatabaseOptimizer`**: Handles Cloud SQL idle/rightsizing.
    -   **`ContainerOptimizer`**: Handles GKE node pools and autoscaling.
    -   **`StorageOptimizer`**: Handles lifecycle policies and snapshot cleanup.
    -   *Each module implements a standard interface: `scan()`, `verify_safe()`, `remediate()`.*

3.  **`FinOptiOrchestrator` (Control Layer)**:
    -   The main agent interface exposed to MCP.
    -   Aggregates results from all modules.
    -   Applies global policies (e.g., "Never auto-delete in PROD").

### 2.2 Data Flow

```mermaid
graph TD
    User[User / MCP Client] -->|1. Run Cost Scan| Orchestrator
    Orchestrator -->|2. Parallel Scan| Compute[Compute Module]
    Orchestrator -->|2. Parallel Scan| SQL[SQL Module]
    Orchestrator -->|2. Parallel Scan| GKE[GKE Module]
    
    Compute -->|3. Get Recommender Data| GCloud[GCloud API]
    SQL -->|3. Get Recommender Data| GCloud
    GKE -->|3. Get Recommender Data| GCloud
    
    Orchestrator -->|4. Return Action Plan| User
    User -->|5. Approve Action| Orchestrator
    Orchestrator -->|6. Execute Remediation| GCloud
```

## 3. Implementation Strategy

### Phase 1: The Unified Scanner (Read-Only)
**Goal**: Build the data gathering capabilities first.
- Implement `get_cost_recommendations.py` logic into a proper Python class structure.
- **Key Deliverable**: A tool `scan_cost_optimizations` that returns a structured JSON report of *all* potential savings across the 4 domains.

### Phase 2: The Remediation Primitives (Write)
**Goal**: Implement the execution logic for each specific type.
- **Compute**: `stop_instance`, `delete_ip`, `snapshot_and_delete_disk`.
- **SQL**: `patch_instance` (stop/resize).
- **GKE**: `resize_cluster`.
- **Key Deliverable**: Validated atomic tools for each action.

### Phase 3: The Safety & Policy Layer
**Goal**: Prevent disasters.
- Implement **Pre-flight Checks**: Before deleting a disk, check `users` list again.
- Implement **Snapshotting**: Before deleting a VM/Disk, force a snapshot.
- Implement **Dry-Run**: A flag `--dry-run` that outputs the exact `gcloud` command without running it.

## 4. MCP Tools Interface

We will expose high-level tools to the AI agent:

1.  `scan_optimization_opportunities(project_id: str, domains: List[str])`
    -   Returns: Comprehensive list of recommendations with savings estimates.

2.  `generate_remediation_plan(recommendation_ids: List[str])`
    -   Returns: A step-by-step plan (natural language + commands) for fixing the selected issues.

3.  `execute_remediation(plan_id: str, confirmation: bool)`
    -   Performs the actual changes.

## 5. Deployment & Testing Plan
- **Mock Testing**: Use recorded API responses to test logic without touching GCP.
- **Dev Verification**: Run against `vector-search-poc` (your current env) to finding real savings.
- **Production Rollout**: Enable `scan` only first, then enable `execute` with approval gates.

This unified approach ensures we don't build 4 separate disjointed tools, but one cohesive "FinOps Agent" that understands the full picture.

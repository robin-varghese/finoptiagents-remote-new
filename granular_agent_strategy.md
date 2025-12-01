# Granular Agentic Strategy for FinOpti Platform (v2: Deeply Nested)

## 1. Executive Summary
We are moving from a "Monolithic Root" to a **Hierarchical Multi-Agent System**. This approach mimics a real-world corporate structure where a "Director" (Root) delegates to "Managers" (Domain Hubs), who in turn delegate to "Specialists" (Leaf Agents).

This specific update focuses on breaking down the **FinOps Analysis** domain into highly specific, task-oriented sub-agents.

## 2. High-Level Architecture
The system is composed of three layers:
1.  **Level 1: The Orchestrator (Root)** - Routes high-level intent (e.g., "Analyze my budget" vs. "Fix this VM").
2.  **Level 2: Domain Managers** - Manage a specific domain (e.g., `FinOpsAnalyticsManager`, `InfrastructureManager`).
3.  **Level 3: Specialist Agents** - Perform the actual atomic work (e.g., `BudgetVarianceAgent`, `VmDeleter`).

## 3. The FinOps Analytics Division (New Breakdown)

Instead of a single "FinOps Analyst," we introduce a **`FinOpsAnalyticsManager`** that oversees five distinct specialists.

### 3.1. `FinOpsAnalyticsManager` (Level 2)
*   **Role:** The "Team Lead". Receives broad analysis requests and delegates to the correct specialist. Aggregates findings to present a summary to the Root.
*   **Tools:** None (Routing only).
*   **Sub-Agents:**
    1.  `BudgetVarianceAgent`
    2.  `ComplianceAuditorAgent`
    3.  `UtilizationAnalystAgent`
    4.  `OptimizationScoutAgent`
    5.  `EnvironmentReadinessAgent`

### 3.2. The Specialists (Level 3)

#### A. `BudgetVarianceAgent`
*   **Mission:** "Follow the Money."
*   **Specific Task:** Compare budgeted costs (`release_train_ticket`) vs. actual costs (`finops_cost_usage`).
*   **Logic:**
    -   Identify projects with >10% variance.
    -   Flag them as "At Risk".
*   **Key Data Sources:** `release_train_ticket` table, `finops_cost_usage` table.

#### B. `ComplianceAuditorAgent`
*   **Mission:** "Enforce the Rules."
*   **Specific Task:** Identify "Rogue Projects".
*   **Logic:**
    -   Check if a project exists in `finops_cost_usage` but is MISSING from `release_train_ticket` OR `earb_review`.
    -   Flag these as "Non-Compliant".
*   **Escalation Path:** Can recommend triggering an EARB review (via `EscalationAgent`).

#### C. `UtilizationAnalystAgent`
*   **Mission:** "Efficiency Check."
*   **Specific Task:** Analyze environment cost ratios and resource idleness.
*   **Logic:**
    -   **Env Check:** Is `Cost(Lower Env) > Cost(Production)`? -> Flag as anomaly.
    -   **Resource Check:** Is `Utilization < 50%`? -> Flag for review.

#### D. `OptimizationScoutAgent`
*   **Mission:** "Find Savings."
*   **Specific Task:** Pinpoint top spenders that are wasteful.
*   **Logic:**
    -   Identify Top 10 cost contributors (Compute, Storage, DBs, Network, Logs).
    -   Cross-reference with utilization data.
    -   Output: "Top Optimization Candidates".

#### E. `EnvironmentReadinessAgent`
*   **Mission:** "Justify Existence."
*   **Specific Task:** Verify if lower environments are actually needed *right now*.
*   **Logic:**
    -   Check `Release Train Tickets` for planned releases.
    -   Check `ServiceNow CR/Defects` for active work.
    -   **Rule:** If a Lower Env exists but has NO active tickets/releases -> Mark as "Zombie Environment" (Optimization Candidate).

## 4. The Action & Escalation Division

Analysis is useless without action. We introduce an **`EscalationAgent`** to handle the "So What?"

### `EscalationAgent` (Level 2 or 3)
*   **Role:** The "Fixer".
*   **Capabilities:**
    -   **ServiceNow:** Create CRs for non-compliant projects or optimization tasks (`create_servicenow_cr`).
    -   **Leadership Alert:** Draft high-priority emails to leadership summarizing the findings (`send_email`).
    -   **EARB Trigger:** Propose/Schedule EARB reviews.
*   **Workflow:** The `FinOpsAnalyticsManager` can pass a list of "Flagged Projects" to the `EscalationAgent` to take bulk action.

## 5. Updated Agent Hierarchy Diagram

```text
Root_Agent (Orchestrator)
├── FinOpsAnalyticsManager
│   ├── BudgetVarianceAgent (Budget vs Actual)
│   ├── ComplianceAuditorAgent (Rogue Projects)
│   ├── UtilizationAnalystAgent (Env Cost Ratio)
│   ├── OptimizationScoutAgent (Top Spenders)
│   └── EnvironmentReadinessAgent (Zombie Envs)
├── InfrastructureManager
│   ├── VmDeleterAgent (Safety Wrapper)
│   └── CpuUtilizationAgent
├── EscalationAgent (ServiceNow, Email)
├── VisualizationAgent (Charts)
└── ComplianceRagAgent (Design Docs)
```

## 6. Implementation Roadmap (Refined)

1.  **Phase 1: The Specialists (SQL & Logic)**
    -   Develop the specific SQL queries for each of the 5 new agents. This is the hardest part—defining the exact logic for "Zombie Environments" or "Budget Variance" in SQL.
2.  **Phase 2: The Manager**
    -   Create `FinOpsAnalyticsManager`. Give it instructions on *when* to call which specialist.
3.  **Phase 3: The Escalator**
    -   Ensure `EscalationAgent` can take structured input (e.g., a JSON list of non-compliant projects) and turn it into a ServiceNow CR or an Email draft.
4.  **Phase 4: Integration**
    -   Wire them up to the Root.

## 7. Benefits of this Deep Granularity
-   **Precision:** The `BudgetVarianceAgent` doesn't need to know about "Zombie Environments". Its context window is 100% focused on math.
-   **Parallelism:** In the future, the Manager could run all 5 specialists in parallel to generate a "Comprehensive Health Report" in seconds.
-   **Auditability:** If a "Zombie Environment" is missed, you know exactly which agent failed (`EnvironmentReadinessAgent`), making debugging trivial.

"""
Pydantic schemas for structured agent outputs.
Ensures type-safe, reliable communication between agents in the granular architecture.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class BudgetVarianceResult(BaseModel):
    """Output schema for the BudgetVarianceAgent."""
    projects_at_risk: list[dict] = Field(
        description="List of projects with >10% budget variance. Each dict contains: project_name, variance_pct, budgeted, actual"
    )
    variance_threshold: float = Field(
        default=0.10,
        description="The variance threshold used for flagging (default 10%)"
    )
    total_projects_analyzed: int = Field(
        description="Total number of projects analyzed"
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Overall severity assessment. low: <15% variance, medium: 15-25%, high: >25%"
    )


class ComplianceResult(BaseModel):
    """Output schema for the ComplianceAuditorAgent."""
    non_compliant_projects: list[str] = Field(
        description="List of project names that are non-compliant"
    )
    violation_type: Literal["missing_from_release_train", "missing_from_earb", "both"] = Field(
        description="Type of compliance violation detected"
    )
    recommended_actions: list[str] = Field(
        description="List of recommended corrective actions"
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level. low: 1-2 projects, medium: 3-5, high: 6-10, critical: >10"
    )


class UtilizationResult(BaseModel):
    """Output schema for the UtilizationAnalystAgent."""
    anomalous_environments: list[dict] = Field(
        description="Environments where lower env cost > production cost. Each dict contains: project, prod_cost, lower_env_cost"
    )
    low_utilization_resources: list[dict] = Field(
        description="Resources with <50% utilization. Each dict contains: resource_id, utilization_pct, cost"
    )
    total_potential_savings: float = Field(
        description="Estimated total monthly savings from addressing inefficiencies"
    )


class OptimizationResult(BaseModel):
    """Output schema for the OptimizationScoutAgent."""
    top_cost_contributors: list[dict] = Field(
        description="Top 10 cost contributors. Each dict contains: resource_type, cost, utilization, savings_potential"
    )
    optimization_candidates: list[str] = Field(
        description="List of resource IDs that are prime optimization targets (high cost + low utilization)"
    )
    estimated_monthly_savings: float = Field(
        description="Estimated monthly savings if all optimization candidates are addressed"
    )


class ReadinessResult(BaseModel):
    """Output schema for the EnvironmentReadinessAgent."""
    zombie_environments: list[dict] = Field(
        description="Lower environments without active justification. Each dict contains: env_name, cost, days_idle, active_tickets"
    )
    justified_environments: list[str] = Field(
        description="List of environment names that have valid justification"
    )
    total_zombie_cost: float = Field(
        description="Total monthly cost of zombie environments"
    )


class FinOpsHealthReport(BaseModel):
    """Output schema for the FinOpsAnalyticsManager."""
    budget_summary: BudgetVarianceResult = Field(
        description="Budget variance analysis results"
    )
    compliance_summary: ComplianceResult = Field(
        description="Compliance audit results"
    )
    utilization_summary: UtilizationResult = Field(
        description="Utilization analysis results"
    )
    optimization_summary: OptimizationResult = Field(
        description="Optimization opportunities"
    )
    readiness_summary: ReadinessResult = Field(
        description="Environment readiness assessment"
    )
    overall_health_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall FinOps health score from 0-100"
    )
    critical_actions_required: list[str] = Field(
        description="Prioritized list of critical actions that need immediate attention"
    )


class AgentErrorResponse(BaseModel):
    """Standardized error response schema for all agents."""
    error_type: Literal["data_unavailable", "query_failed", "api_timeout"] = Field(
        description="Category of error encountered"
    )
    error_message: str = Field(
        description="Detailed error message explaining what went wrong"
    )
    fallback_executed: bool = Field(
        description="Whether a fallback strategy was executed"
    )
    partial_results: Optional[dict] = Field(
        default=None,
        description="Any partial results that were obtained before the error"
    )


class VmDeletionAuditResult(BaseModel):
    """Output schema for the VmDeletionAuditorAgent."""
    deleted_vms: list[dict] = Field(
        description="List of deleted VMs. Each dict contains: vm_name, deleted_by, deletion_timestamp, zone, project_id"
    )
    total_deletions: int = Field(
        description="Total number of VM deletions found"
    )
    query_timeframe: str = Field(
        description="Timeframe of the query (e.g., 'today', 'yesterday', 'last 7 days', 'all time')"
    )
    queried_user: Optional[str] = Field(
        default=None,
        description="Username filter if query was filtered by specific user"
    )


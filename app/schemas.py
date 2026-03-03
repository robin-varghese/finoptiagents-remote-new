"""
Pydantic schemas for structured agent outputs.
Ensures type-safe, reliable communication between agents in the granular architecture.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


from pydantic import BaseModel, Field
from typing import Literal, Optional


class ProjectVariance(BaseModel):
    """Specific variance details for a project."""
    project_name: str = Field(description="Name of the project")
    variance_pct: float = Field(description="Budget variance percentage")
    budgeted: float = Field(description="Budgeted cost amount")
    actual: float = Field(description="Actual spent cost amount")


class BudgetVarianceResult(BaseModel):
    """Output schema for the BudgetVarianceAgent."""
    projects_at_risk: list[ProjectVariance] = Field(
        description="List of projects with >10% budget variance."
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


class AnomalousEnvironment(BaseModel):
    """Details of environment cost anomalies."""
    project: str = Field(description="Project name")
    prod_cost: float = Field(description="Production environment cost")
    lower_env_cost: float = Field(description="Lower environment cost")


class LowUtilizationResource(BaseModel):
    """Details of low-utilization resources."""
    resource_id: str = Field(description="Unique identifier for the resource")
    utilization_pct: float = Field(description="Current utilization percentage")
    cost: float = Field(description="Monthly cost of the resource")


class UtilizationResult(BaseModel):
    """Output schema for the UtilizationAnalystAgent."""
    anomalous_environments: list[AnomalousEnvironment] = Field(
        description="Environments where lower env cost > production cost."
    )
    low_utilization_resources: list[LowUtilizationResource] = Field(
        description="Resources with <50% utilization."
    )
    total_potential_savings: float = Field(
        description="Estimated total monthly savings from addressing inefficiencies"
    )


class CostContributor(BaseModel):
    """Details of a major cost-contributing resource."""
    resource_type: str = Field(description="Type of resource (e.g., Compute, Storage)")
    cost: float = Field(description="Monthly cost")
    utilization: float = Field(description="Average utilization percentage")
    savings_potential: float = Field(description="Estimated monthly savings potential")


class OptimizationResult(BaseModel):
    """Output schema for the OptimizationScoutAgent."""
    top_cost_contributors: list[CostContributor] = Field(
        description="Top 10 cost contributors."
    )
    optimization_candidates: list[str] = Field(
        description="List of resource IDs that are prime optimization targets (high cost + low utilization)"
    )
    estimated_monthly_savings: float = Field(
        description="Estimated monthly savings if all optimization candidates are addressed"
    )


class ZombieEnvironment(BaseModel):
    """Details of potentially redundant environment."""
    env_name: str = Field(description="Environment name")
    cost: float = Field(description="Monthly cost")
    days_idle: int = Field(description="Number of days since last activity")
    active_tickets: int = Field(description="Number of active tickets related to this env")


class ReadinessResult(BaseModel):
    """Output schema for the EnvironmentReadinessAgent."""
    zombie_environments: list[ZombieEnvironment] = Field(
        description="Lower environments without active justification."
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


class PartialErrorResult(BaseModel):
    """Partial results obtained before an error occurred."""
    message: str = Field(description="Summary of partial progress")
    data: Optional[str] = Field(default=None, description="Optional partial data captured")


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
    partial_results: Optional[PartialErrorResult] = Field(
        default=None,
        description="Any partial results that were obtained before the error"
    )


class DeletedVm(BaseModel):
    """Audit details for a deleted VM."""
    vm_name: str = Field(description="Name of the VM")
    deleted_by: str = Field(description="User who performed the deletion")
    deletion_timestamp: str = Field(description="Timestamp of the deletion event")
    zone: str = Field(description="GCP zone")
    project_id: str = Field(description="GCP project ID")


class VmDeletionAuditResult(BaseModel):
    """Output schema for the VmDeletionAuditorAgent."""
    deleted_vms: list[DeletedVm] = Field(
        description="List of deleted VMs."
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


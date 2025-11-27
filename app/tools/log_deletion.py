from google.adk.tools import FunctionTool, ToolContext
from .log_vm_deletion_to_bigquery import log_vm_deletion_to_bigquery

def log_deletion_tool_wrapper(project_id: str, instance_id: str, zone: str, tool_context: ToolContext) -> str:
    """
    A wrapper for the log_vm_deletion_to_bigquery function to be used as an ADK tool.
    It extracts the user_id from the tool_context and calls the logging function.
    """
    user_id = tool_context.state.get("user_id")
    if not user_id:
        return "Error: user_id not found in tool_context."
    
    return log_vm_deletion_to_bigquery(
        project_id=project_id,
        instance_id=instance_id,
        zone=zone,
        user_id=user_id,
        tool_context=tool_context
    )

log_deletion_tool = FunctionTool(
    func=log_deletion_tool_wrapper,
    description="Logs the deletion of a VM instance to BigQuery."
)

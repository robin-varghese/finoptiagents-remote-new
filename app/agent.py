import nest_asyncio
# =======================================================================================
# ### --- START: COMPLETE AGENT FILE (DEFINITIVE SIMPLIFIED SOLUTION) --- ###
# # =======================================================================================
# 1. --- IMPORTS ---
import logging
from typing import Optional

from google.adk.agents import LlmAgent, Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from . import descandinstructions
from . import schemas
from app import config

logger = logging.getLogger(__name__)

# --- Direct Tool Imports (Bypassing MCP Subprocess) ---
# Importing tools directly from mcp_server to avoid Streamlit/asyncio/subprocess compatibility issues
from mcp_server.tools import (
    run_bq_query,
    send_email,
    list_vm_instances,
    generate_chart_from_data,
    call_cpu_utilization_agent,
    rag_query,
    create_corpus,
    add_data,
    delete_corpus,
    delete_document,
    get_corpus_info,
    get_corpus_info,
    list_corpora,
    log_savings_impact,
    scan_cost_recommendations
)
from app.tools.gcloud_mcp_tools import run_gcloud_command
from app.tools.monitoring_mcp_tools import query_time_series, query_logs, list_metrics



# List of all available tools
ALL_TOOLS = [
    run_bq_query,
    send_email,
    list_vm_instances,
    generate_chart_from_data,
    call_cpu_utilization_agent,
    rag_query,
    create_corpus,
    add_data,
    delete_corpus,
    delete_document,
    get_corpus_info,
    list_corpora,
    log_savings_impact,
    scan_cost_recommendations
]


#*************************START: Call Back ***************************************
def simple_before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Inspects/modifies the LLM request or skips the call.
    This version safely handles system_instruction as a str or Content object.
    """
    agent_name = callback_context.agent_name
    logging.info(f"Before model call for agent: {agent_name}")

    prefix = "[Modified by Callback] "
    current_instruction = llm_request.config.system_instruction

    base_text = ""
    if isinstance(current_instruction, str):
        base_text = current_instruction
    elif isinstance(current_instruction, types.Content) and current_instruction.parts:
        base_text = current_instruction.parts[0].text or ""

    modified_text = prefix + base_text
    llm_request.config.system_instruction = types.Content(
        role="system",
        parts=[types.Part(text=modified_text)]
    )
    logging.debug(f"Modified system instruction for '{agent_name}'.")

    last_user_message = ""
    if llm_request.contents:
        last_content_item = llm_request.contents[-1]
        if last_content_item.role == 'user' and last_content_item.parts:
            if last_content_item.parts[0].text is not None:
                last_user_message = last_content_item.parts[0].text

    logging.info(f"Inspecting last user message for '{agent_name}': '{last_user_message}'")

    if "BLOCK" in last_user_message.upper():
        logging.warning("'BLOCK' keyword found in user message. Skipping LLM call.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    else:
        logging.info(f"Proceeding with LLM call for agent '{agent_name}'.")
        return None


#*************************END: Call Back *****************************************

# =============================================================================
# FinOps Specialist Agents (Granular Architecture)
# =============================================================================

# --- Budget Variance Agent ---
budget_variance_agent = LlmAgent(
    name="budget_variance_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.budget_variance_agent_description,
    instruction=descandinstructions.budget_variance_agent_instruction,
    tools=[run_bq_query],
    output_schema=schemas.BudgetVarianceResult,
)

# --- Compliance Auditor Agent ---
compliance_auditor_agent = LlmAgent(
    name="compliance_auditor_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.compliance_auditor_agent_description,
    instruction=descandinstructions.compliance_auditor_agent_instruction,
    tools=[run_bq_query],
    output_schema=schemas.ComplianceResult,
)

# --- Utilization Analyst Agent ---
utilization_analyst_agent = LlmAgent(
    name="utilization_analyst_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.utilization_analyst_agent_description,
    instruction=descandinstructions.utilization_analyst_agent_instruction,
    tools=[run_bq_query],
    output_schema=schemas.UtilizationResult,
)

# --- Optimization Scout Agent (BigQuery-based) ---
optimization_scout_agent = LlmAgent(
    name="optimization_scout_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.optimization_scout_agent_description,
    instruction=descandinstructions.optimization_scout_agent_instruction,
    tools=[run_bq_query],
    output_schema=schemas.OptimizationResult,
)

# --- Environment Readiness Agent ---
environment_readiness_agent = LlmAgent(
    name="environment_readiness_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.environment_readiness_agent_description,
    instruction=descandinstructions.environment_readiness_agent_instruction,
    tools=[run_bq_query],
    output_schema=schemas.ReadinessResult,
)

# --- GCloud Recommender Agent (NEW) ---
gcloud_recommender_agent = LlmAgent(
    name="gcloud_recommender_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.gcloud_recommender_agent_description,
    instruction=descandinstructions.gcloud_recommender_agent_instruction,
    tools=[run_gcloud_command, scan_cost_recommendations],
)

# --- FinOps Analytics Manager (Coordinates all specialists) ---
finops_analytics_manager = LlmAgent(
    name="finops_analytics_manager",
    model=config.FINOPTIAGENTS_LLM,  # Using Pro for better aggregation capabilities
    description=descandinstructions.finops_analytics_manager_description,
    instruction=descandinstructions.finops_analytics_manager_instruction,
    sub_agents=[
        budget_variance_agent,
        compliance_auditor_agent,
        utilization_analyst_agent,
        optimization_scout_agent,
        environment_readiness_agent,
    ],
    output_schema=schemas.FinOpsHealthReport,
)

# --- Escalation Agent (Converts findings to actions) ---
escalation_agent = LlmAgent(
    name="escalation_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.escalation_agent_description,
    instruction=descandinstructions.escalation_agent_instruction,
    tools=[send_email],  # ServiceNow integration to be added later
)

# --- Log Savings Impact Agent (NEW) ---
log_savings_impact_agent = LlmAgent(
    name="log_savings_impact_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.compliance_logger_agent_description, # Keep old description for now
    instruction=descandinstructions.compliance_logger_agent_instruction, # Keep old instruction for now
    tools=[log_savings_impact],
)

# --- Visualization Agent (NEW) ---
visualization_agent = LlmAgent(
    name="visualization_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.visualization_agent_description,
    instruction=descandinstructions.visualization_agent_instruction,
    tools=[run_bq_query, generate_chart_from_data],
)

# =============================================================================
# Existing Agents (Retained from previous architecture)
# =============================================================================



greeting_agent = LlmAgent(
    name="Greeter",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.greeting_agent_description,
    instruction=descandinstructions.greeting_agent_instruction
)

gcloud_ops_agent = LlmAgent(
    name="gcloud_ops_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.gcloud_ops_agent_description,
    instruction=descandinstructions.gcloud_ops_agent_instruction,
    tools=[run_gcloud_command],
)

monitoring_agent = LlmAgent(
    name="monitoring_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.monitoring_agent_description,
    instruction=descandinstructions.monitoring_agent_instruction,
    tools=[query_time_series, query_logs, list_metrics],
)

# --- The Single, Simplified, and Robust RAG Agent ---
# --- CORRECTED DEBUGGING CALLBACK ---
def debug_after_model(callback_context, llm_response):
    """
    This callback function will intercept and print the raw response from the LLM,
    allowing us to see the exact tool call it is trying to make.
    """
    logging.debug("="*50)
    logging.debug(f"INTERCEPTING MODEL RESPONSE for agent: {callback_context.agent_name}")
    # The llm_response object contains the model's output.
    # We are interested in the tool_calls part.
    logging.debug("--- RAW LLM Response ---")
    logging.debug(llm_response)
    logging.debug("="*50)





# Tools available to the Root Agent (Subset of ALL_TOOLS)
# We exclude specialized tools like 'scan_cost_recommendations' to FORCE delegation to specialists.
ROOT_TOOLS = [
    run_bq_query,
    send_email,
    list_vm_instances,
    # generate_chart_from_data, # REMOVED: Delegated to visualization_agent
    call_cpu_utilization_agent,
    # rag_query, # Root delegates RAG to rag_agent
    # create_corpus,
    # add_data,
    # delete_corpus,
    # delete_document,
    # get_corpus_info,
    # list_corpora,
    log_savings_impact,
    run_gcloud_command, # Kept for "General Inventory" capability
    # scan_cost_recommendations # REMOVED: Must delegate to gcloud_recommender_agent
]

# --- BQ Auditor Agent (NEW) ---
bq_auditor_agent = LlmAgent(
    name="bq_auditor_agent",
    model=config.FINOPTIAGENTS_LLM,
    description=descandinstructions.bq_auditor_agent_description,
    instruction=descandinstructions.bq_auditor_agent_instruction,
    tools=[run_bq_query],
)


try:
    design_compliance_check_rag_agent = Agent(
        name="design_compliance_check_rag_agent",
        # Using Gemini 2.5 Flash for best performance with RAG operations
        model=config.FINOPTIAGENTS_LLM,
        description=descandinstructions.rag_agent_description,
        tools=ALL_TOOLS,  # Direct tool imports (no MCP subprocess)
        instruction=descandinstructions.rag_agent_instruction,
    )

    # --- Final, Simplified Root Agent ---
    root_agent = LlmAgent(
        name="finops_optimization_agent",
        # IMPORTANT: Bidirectional streaming requires a model that supports this feature,
        # often a "Live" or "Express" version. The name below is an example;
        # you must use the specific model name provided for this capability.
        model=config.FINOPTIAGENTS_LLM,
        description=descandinstructions.root_agent_description,
        instruction=(descandinstructions.root_agent_instruction),
        # --- SOLUTION: Move the agent from 'tools' to 'sub_agents' ---
        tools=ROOT_TOOLS,  # Direct tool imports (no MCP subprocess)
        sub_agents=[
            # FinOps Manager (for bulk operations and routes to specialists internally)
            finops_analytics_manager,
            # Action agents
            escalation_agent,
            # Compliance agents
            log_savings_impact_agent,
            greeting_agent,
            design_compliance_check_rag_agent,
            gcloud_ops_agent,
            monitoring_agent,
            gcloud_recommender_agent, # Added the new agent
            bq_auditor_agent, # Added the new agent
            visualization_agent, # Added the new agent
        ],
        before_model_callback=simple_before_model_modifier,
    )

    # --- App Container with Plugins ---
    from google.adk.apps.app import App
    from google.adk.plugins import ReflectAndRetryToolPlugin
    from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin

    # --- Initialize and attach the new analytics plugin ---
    analytics_plugin = BigQueryAgentAnalyticsPlugin(
        project_id=config.GOOGLE_PROJECT_ID,
        dataset_id=config.BIGQUERY_DATASET_ID,
        table_id=config.BIGQUERYAGENTANALYTICSPLUGIN_TABLE_ID,
        location=config.GOOGLE_ZONE
    )

    finops_app = App(
        name="finoptiagents_app",
        root_agent=root_agent,
        plugins=[
            ReflectAndRetryToolPlugin(max_retries=3),
            analytics_plugin,
        ]
    )

    logger.info("Agent and App created successfully.")
except Exception as e:
    logger.error(f"Failed to create agent: {e}")
    raise
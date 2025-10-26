import json
import logging

from google.cloud import bigquery

from .. import config

def run_bq_query(query: str) -> str:
    """
    Executes a read-only BigQuery SQL query against the configured GCP project and returns the results.

    This is your primary tool for understanding the state of the cloud environment.
    You do NOT need to specify a project_id; the tool runs in the correct project automatically.
    All the table attributes are set with descriptions. So chech the description of columns to identify the correct columns and make correct queries.

    The table names in your query, like `finoptiagents.finops_cost_usage`, already contain the dataset.
    ***For exclusive requets on VM Deletion Logs
    Route the requests specific for VM deletion scenarion to table finops_agent_logs.vm_deletion_log
    ***
    For any other queries use the following ables in vector-search-poc.finoptiagents dataset
    1. project_information_master: Core project details. This is the central registry of all projects. 
    Use it to find project names, owners, and IDs. The stakeholder details are mentioned in this table, product_owner_name, business_service_owner_name
    2. project_information_child: Individual cloud resources. This table contains a detailed inventory of every single provisioned resource 
    in a project(like VMs, databases, etc.) for each project.
    3. finops_cost_usage: Raw monthly cost data with environment break-up. This table holds the raw financial and performance metrics. Use it for detailed 
    analysis of monthly costs and resource utilization percentages.
    4. servicenow_change_defect: Development tickets. This table tracks active development and bug fixes from ServiceNow. A project with open tickets here is 
    considered "active," justifying its operational costs. The entry made for a project is irrespective of the environment. In other words, a ticket raised for 
    dev environment is impacting the enite project
    5. earb_review: Governance approvals. This table logs which projects have passed the formal Enterprise Architecture Review Board (EARB) process. A missing 
    entry here is a major governance red flag.
    6. release_train_ticket: Release planning. This table lists projects that are officially part of a planned software release train. The project budget is 
    stored in this table. An entry made for a project, irrespective of environment is considered to be part of the release train.

    A project not in this list may be unauthorized or "shadow IT." 
    
    Common Analysis
    Budgeted & actual cost spent analysis: by comparing the budgeted cost in release_train_ticket and the actual cost in finops_cost_usage, 
    this can be identifyed. Ideally the projects spending near (10% varience) to the budgeted cost is a good project. Otherwise its a bad project
    Non-Compliance Analysis: The projects which were not part of release_train_ticket and/or earb_review can be onsidered as non-compliance and bad projects.
    Projects which are Non-Compliant, escalate this to leadership team. Trigger EARB review for resources exemption from automated optimization; 
    open ServiceNow CR with full analysis and route to stakeholders for approval.
    Utilisation Analysis: The projects burning more for their lower environemts than production environment can be considered as bad projects.
    The projects were the resource utilization is low is also considered as bad projects. table finops_cost_usage has this info.
    Optimization Analysis: Identify top cost-contributing resources with optimization chances (compute, storage, managed DBs, networking egress, 
    logging/monitoring). Also highlight the resources where utlization is less than 50%.
    Readiness Check for Lower Environments: Cross-check Release Train Tickets and ServiceNow CR/Defects to confirm upcoming releases or open CRs. 
    If there are no planned release then there is no point to have lower environment. Mark such lower-env resources as optimization candidates.
    """
    logging.info(f"Executing read-only BigQuery query.")
    logging.debug(f"BQ Query: {query}")
    if not config.GOOGLE_PROJECT_ID:
        return json.dumps({"error": "Configuration error: GOOGLE_PROJECT_ID is not set."})
    try:
        if any(keyword in query.upper() for keyword in ['INSERT', 'UPDATE', 'DELETE', 'MERGE', 'TRUNCATE', 'CREATE', 'DROP', 'ALTER']):
            return json.dumps({"error": "This tool is for read-only SELECT queries."})
        client = bigquery.Client(project=config.GOOGLE_PROJECT_ID)
        results = client.query(query).result()
        if results.total_rows == 0:
            return json.dumps({"total_rows_found": 0, "data_sample": []})
        data_sample = [dict(row) for i, row in enumerate(results) if i < 25]
        return json.dumps({"total_rows_found": results.total_rows, "rows_returned_in_sample": len(data_sample), "data_sample": data_sample}, default=str)
    except Exception as e:
        logging.error(f"BigQuery query execution failed: {e}", exc_info=True)
        return json.dumps({"error": f"An error occurred while running the query: {str(e)}"})

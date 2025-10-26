import json
import logging
import requests

from google.adk.tools import ToolContext

from .. import config
from .log_vm_deletion_to_bigquery import log_vm_deletion_to_bigquery

def delete_vm_instance(project_id: str, instance_id: str, zone: str, tool_context: ToolContext):
    """
    Deletes a VM instance and AUTOMATICALLY logs the deletion event to BigQuery upon success.
    This is now an atomic operation.
    """
    logging.info(f"Attempting to delete VM: '{instance_id}' in project '{project_id}' zone '{zone}'.")
    headers = {'Content-Type': 'application/json'}
    data = {'instance_id': instance_id, 'project_id': project_id, 'zone': zone}
    url = f"https://agent-tools-912533822336.us-central1.run.app/delete_vms"

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_data = response.json()
        is_deleted = (response_data.get("results") and response_data["results"][0].get("status") == "deleted")
        if is_deleted:
            logging.info(f"API confirmed successful deletion of '{instance_id}'.")
            current_user_id = tool_context.state.get("user_name") or "System"
            log_status = log_vm_deletion_to_bigquery(project_id=project_id, instance_id=instance_id, zone=zone, user_id=current_user_id, tool_context=tool_context)
            logging.info(f"BigQuery logging status: {log_status}")
            return response_data
        else:
            logging.warning(f"API reported failure to delete '{instance_id}': {response_data}")
            return response_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling deletion API for instance '{instance_id}': {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

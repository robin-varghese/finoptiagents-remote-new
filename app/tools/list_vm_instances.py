import json
import logging
import requests

from .. import config

def list_vm_instances(project_id: str, zone: str):
    """Lists VM instances based on domain, project ID, and zone."""
    logging.info(f"Listing VM instances for project '{project_id}' in zone '{zone}'.")
    headers = {'Content-Type': 'application/json'}
    data = {'project_id': project_id, 'zone': zone}
    url = "https://agent-tools-912533822336.us-central1.run.app/list_vms"
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error listing instances for project '{project_id}': {e}", exc_info=True)
        return None

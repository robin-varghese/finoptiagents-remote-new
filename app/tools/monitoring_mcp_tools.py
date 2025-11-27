import os
import json
import logging
from typing import List, Optional, Any, Dict
from mcp_server.external.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# Configuration
DOCKER_IMAGE = "gcloud-monitoring-mcp-image"
MOUNT_PATH = f"{os.path.expanduser('~')}/.config/gcloud:/root/.config/gcloud"

class MonitoringMCPClient(MCPClient):
    """
    Specialized client for the Monitoring MCP server.
    """
    def __init__(self):
        super().__init__(
            docker_image=DOCKER_IMAGE,
            mount_path=MOUNT_PATH
        )

async def query_time_series(project_id: str, metric_type: str, resource_filter: str, minutes_ago: int) -> Dict[str, Any]:
    """
    Queries time series data from Cloud Monitoring.

    Args:
        project_id: The GCP project ID.
        metric_type: The metric type (e.g., 'compute.googleapis.com/instance/cpu/utilization').
        resource_filter: The resource filter (e.g., 'resource.labels.instance_id="123"').
        minutes_ago: How many minutes of history to retrieve.

    Returns:
        A dictionary containing the time series data.
    """
    logger.info(f"Querying time series: {metric_type} for {project_id}")
    async with MonitoringMCPClient() as client:
        try:
            result_str = await client.call_tool(
                "query_time_series",
                arguments={
                    "project_id": project_id,
                    "metric_type": metric_type,
                    "resource_filter": resource_filter,
                    "minutes_ago": minutes_ago
                }
            )
            return json.loads(result_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from query_time_series: {e}")
            return {"error": "Invalid JSON response from server", "details": str(e)}
        except Exception as e:
            logger.error(f"Failed to query time series: {e}")
            return {"error": f"Error querying time series: {str(e)}"}

async def query_logs(project_id: str, filter: str, hours_ago: int = 1, limit: int = 20) -> Dict[str, Any]:
    """
    Queries logs from Cloud Logging.

    Args:
        project_id: The GCP project ID.
        filter: The advanced logs filter (e.g., 'severity>=ERROR').
        hours_ago: How many hours of logs to search.
        limit: Maximum number of log entries to return.

    Returns:
        A dictionary containing the log entries.
    """
    logger.info(f"Querying logs: {filter} for {project_id}")
    async with MonitoringMCPClient() as client:
        try:
            result_str = await client.call_tool(
                "query_logs",
                arguments={
                    "project_id": project_id,
                    "filter": filter,
                    "hours_ago": hours_ago,
                    "limit": limit
                }
            )
            return json.loads(result_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from query_logs: {e}")
            return {"error": "Invalid JSON response from server", "details": str(e)}
        except Exception as e:
            logger.error(f"Failed to query logs: {e}")
            return {"error": f"Error querying logs: {str(e)}"}

async def list_metrics(project_id: str, filter: str = "") -> Dict[str, Any]:
    """
    Lists available metric descriptors.

    Args:
        project_id: The GCP project ID.
        filter: Optional filter to narrow down metrics.

    Returns:
        A dictionary containing the list of metrics.
    """
    logger.info(f"Listing metrics for {project_id}")
    async with MonitoringMCPClient() as client:
        try:
            result_str = await client.call_tool(
                "list_metrics",
                arguments={
                    "project_id": project_id,
                    "filter": filter
                }
            )
            return json.loads(result_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from list_metrics: {e}")
            return {"error": "Invalid JSON response from server", "details": str(e)}
        except Exception as e:
            logger.error(f"Failed to list metrics: {e}")
            return {"error": f"Error listing metrics: {str(e)}"}

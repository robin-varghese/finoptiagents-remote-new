# FinOptiAgents Feature Specification

This document outlines the exhaustive list of cost optimization capabilities for **FinOptiAgents**. The agent will autonomously detect and remediate cost inefficiencies using the **Google Cloud MCP Server**.

## 1. Compute Engine Optimization

### 1.1 Idle VM Deletion
**Description**: Identify and delete (or stop) virtual machines that have been idle for a specific period.
- **Recommender ID**: `google.compute.instance.IdleResourceRecommender`
- **Detection**: Check for low CPU/Network usage over 14 days.
- **Action**:
  - *Stop*: `gcloud compute instances stop [INSTANCE_NAME] --zone [ZONE]`
  - *Delete*: `gcloud compute instances delete [INSTANCE_NAME] --zone [ZONE]` (Optionally snapshot first)

### 1.2 VM Rightsizing (Downsizing/Upsizing)
**Description**: Resize VMs to match their actual workload requirements.
- **Recommender ID**: `google.compute.instance.MachineTypeRecommender`
- **Detection**: Analyze CPU and Memory utilization.
- **Action**:
  1. Stop Instance: `gcloud compute instances stop [INSTANCE_NAME] --zone [ZONE]`
  2. Set Machine Type: `gcloud compute instances set-machine-type [INSTANCE_NAME] --machine-type [NEW_TYPE] --zone [ZONE]`
  3. Start Instance: `gcloud compute instances start [INSTANCE_NAME] --zone [ZONE]`

### 1.3 Unused IP Address Release
**Description**: Release static external IP addresses that are reserved but not attached to any resource.
- **Recommender ID**: `google.compute.address.IdleResourceRecommender`
- **Detection**: IPs with `status=RESERVED` but no `users` list.
- **Action**: `gcloud compute addresses delete [ADDRESS_NAME] --region [REGION]` or `--global`

### 1.4 Idle Persistent Disk Deletion
**Description**: Delete or snapshot unattached persistent disks.
- **Recommender ID**: `google.compute.disk.IdleResourceRecommender`
- **Detection**: Disks not attached to any VM for a set period.
- **Action**: `gcloud compute disks delete [DISK_NAME] --zone [ZONE]` (Optionally verify `users` list is empty)

## 2. Managed Services Optimization

### 2.1 Cloud SQL Idle Instance Stop
**Description**: Stop Cloud SQL instances that show no activity (e.g., dev/test environments left running).
- **Recommender ID**: `google.cloudsql.instance.IdleRecommender`
- **Detection**: Low connections/CPU.
- **Action**: `gcloud sql instances patch [INSTANCE_NAME] --activation-policy=NEVER` (Stops the instance)

### 2.2 Cloud SQL Rightsizing
**Description**: Downsize over-provisioned Cloud SQL instances (CPU/RAM).
- **Recommender ID**: `google.cloudsql.instance.OverprovisionedRecommender`
- **Action**: `gcloud sql instances patch [INSTANCE_NAME] --tier [NEW_TIER]`

### 2.3 GKE Cluster Rightsizing & Autoscaling
**Description**: Optimize Kubernetes clusters by resizing node pools or enabling autoscaling.
- **Recommender ID**: `google.container.DiagnosisRecommender` (and others specific to utilization)
- **Actions**:
  - *Resize Pool*: `gcloud container clusters resize [CLUSTER_NAME] --node-pool [POOL_NAME] --num-nodes [NUM] --zone [ZONE]`
  - *Enable Autoscaling*: `gcloud container clusters update [CLUSTER_NAME] --enable-autoscaling ...`

## 3. Storage & Cleanup

### 3.1 Cloud Storage Lifecycle Management
**Description**: Move infrequently accessed objects to cheaper storage classes (Nearline/Coldline/Archive).
- **Tooling**: Often rule-based rather than direct "recommender" API, but agent can configure buckets.
- **Action**: `gcloud storage buckets update gs://[BUCKET_NAME] --lifecycle-file [JSON_CONFIG]`

### 3.2 Old Snapshot Cleanup
**Description**: Delete disk snapshots older than X days (retention policy enforcement).
- **Action**: List snapshots with filter `creationTimestamp < DATE`, then `gcloud compute snapshots delete [SNAPSHOT_NAME]`

## 4. Architectural Summary for FinOptiAgents

The FinOptiAgents system should operate in a detection-response loop:

1.  **Scan**: Query `gcloud recommender` APIs for pending recommendations.
2.  **Filter**: Apply policy rules (e.g., "Auto-delete idle IPs in dev project", "Ask approval for Prod VM deletion").
3.  **Act**: Execute the corresponding `gcloud` command via the **MCP Server**.
4.  **Verify**: Confirm the resource is modified/deleted and the recommendation state is updated.

This implementation plan provides the blueprint for integrating these features into your agent.

import logging
import subprocess
import os
import re
from dotenv import dotenv_values
from google.cloud import secretmanager
from google.api_core import exceptions
from google.protobuf import field_mask_pb2

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Helper Functions ---

def get_gcloud_project_id() -> str:
    """Gets the current project ID from the gcloud CLI configuration."""
    try:
        project_id = subprocess.run(
            ['gcloud', 'config', 'get-value', 'project'],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        if not project_id:
            raise ValueError("gcloud project ID is not set.")
        logging.info(f"Successfully detected gcloud project: {project_id}")
        return project_id
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logging.error(
            "Could not get project ID from gcloud. Please ensure 'gcloud' is installed, "
            "authenticated, and a default project is set via 'gcloud config set project YOUR_PROJECT_ID'."
        )
        raise SystemExit(e)

def sanitize_for_label(text: str) -> str:
    """Sanitizes a string to be a valid Google Cloud label value."""
    # Convert to lowercase
    text = text.lower()
    # Replace invalid characters with a hyphen
    text = re.sub(r'[^a-z0-9-]', '-', text)
    # Must not be longer than 63 chars, start/end with a letter or number
    text = text.strip('-')[:63]
    return text

def upsert_secret(client: secretmanager.SecretManagerServiceClient, project_id: str, secret_name: str, secret_value: str, desired_labels: dict):
    """
    Creates/updates a secret and its labels, then creates/updates its value version.
    """
    parent = f"projects/{project_id}"
    secret_path = f"{parent}/secrets/{secret_name}"
    secret_exists = True

    try:
        # Step 1: Check secret metadata (existence and labels)
        logging.info(f"Checking for existing secret metadata: '{secret_name}'")
        secret_obj = client.get_secret(request={"name": secret_path})

        # Compare labels and update if necessary
        if secret_obj.labels != desired_labels:
            logging.warning(f"Labels for '{secret_name}' are outdated. Updating.")
            secret_obj.labels.clear()
            secret_obj.labels.update(desired_labels)
            
            update_mask = field_mask_pb2.FieldMask(paths=["labels"])
            client.update_secret(request={"secret": secret_obj, "update_mask": update_mask})
            logging.info(f"Successfully updated labels for '{secret_name}'.")
        else:
            logging.info(f"Labels for '{secret_name}' are already correct.")

    except exceptions.NotFound:
        # Secret does not exist, so create it with the correct labels
        secret_exists = False
        logging.info(f"Secret '{secret_name}' not found. Creating it with labels.")
        try:
            client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_name,
                    "secret": {
                        "replication": {"automatic": {}},
                        "labels": desired_labels
                    },
                }
            )
            logging.info(f"Successfully created secret container for '{secret_name}'.")
        except exceptions.AlreadyExists:
             logging.warning(f"Secret '{secret_name}' was created by another process. Proceeding.")
             secret_exists = True

    # Step 2: Check the secret value and add a new version if needed
    if secret_exists:
        try:
            latest_version_path = f"{secret_path}/versions/latest"
            response = client.access_secret_version(request={"name": latest_version_path})
            latest_value = response.payload.data.decode("UTF-8")

            if latest_value == secret_value:
                logging.info(f"Value for '{secret_name}' is already up-to-date. Skipping version add.")
                return
            else:
                logging.warning(f"Value for '{secret_name}' has changed. Adding a new version.")
                add_secret_version(client, secret_path, secret_value)
        except exceptions.NotFound:
            # Secret exists but has no versions. This is a rare edge case.
            logging.warning(f"Secret '{secret_name}' exists but has no versions. Adding first version.")
            add_secret_version(client, secret_path, secret_value)
    else:
        # The secret was just created, so it definitely needs its first version
        add_secret_version(client, secret_path, secret_value)

def add_secret_version(client: secretmanager.SecretManagerServiceClient, secret_path: str, secret_value: str):
    """Adds a new version to an existing secret."""
    try:
        payload = secret_value.encode("UTF-8")
        client.add_secret_version(
            request={"parent": secret_path, "payload": {"data": payload}}
        )
        secret_name = secret_path.split('/')[-1]
        logging.info(f"Successfully added new version to secret '{secret_name}'.")
    except exceptions.GoogleAPICallError as e:
        logging.error(f"Failed to add a new version to {secret_path}: {e}")
        raise

# --- Main Execution Logic ---

def main():
    """Main function to run the secret upsert process."""
    try:
        project_id = get_gcloud_project_id()
        client = secretmanager.SecretManagerServiceClient()

        # --- Gather Tags (Labels) ---
        folder_name = sanitize_for_label(os.path.basename(os.getcwd()))
        
        app_name = ""
        while not app_name:
            app_name_input = input("Please enter a project/application name to use as a tag (e.g., 'customer-portal'): ")
            if app_name_input.strip():
                app_name = sanitize_for_label(app_name_input)
            else:
                print("Application name cannot be empty. Please try again.")

        labels = {
            "folder-source": folder_name,
            "application": app_name
        }
        logging.info(f"Using the following labels for all secrets: {labels}")

        # --- Load and Process .env file ---
        env_path = ".env"
        if not os.path.exists(env_path):
            logging.error(f"The '{env_path}' file was not found in the current directory.")
            return

        secrets_to_upload = dotenv_values(env_path)
        if not secrets_to_upload:
            logging.warning(f"The '{env_path}' file is empty. No secrets to process.")
            return

        logging.info(f"Found {len(secrets_to_upload)} secrets in '{env_path}' to process.")
        print("-" * 30) # Separator for readability

        for secret_name, secret_value in secrets_to_upload.items():
            if not secret_value:
                logging.warning(f"Skipping '{secret_name}' because its value is empty.")
                continue
            
            # Google Secret Manager names must follow specific rules.
            sanitized_name = sanitize_for_label(secret_name)
            if sanitized_name != secret_name.lower():
                logging.info(f"Sanitizing secret name from '{secret_name}' to '{sanitized_name}'")

            upsert_secret(client, project_id, sanitized_name, secret_value, labels)
            print("-" * 30) # Separator for each secret

        logging.info("--- Secret synchronization complete. ---")

    except exceptions.PermissionDenied:
        logging.error(
            "PERMISSION DENIED: The authenticated user/service account is missing IAM permissions. "
            "Please grant the 'Secret Manager Admin' (roles/secretmanager.admin) role to the user."
        )
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
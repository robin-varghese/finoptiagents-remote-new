import json
import logging
import requests

from google.adk.tools import ToolContext

def send_email(to_address: str, subject: str, user_name: str, body_html_base64: str, tool_context: ToolContext) -> dict:
    """Sends an email to the specified recipient.

    Args:
        to_address (str): The recipient's email address.
        subject (str): The subject of the email.
        user_name (str): The name of the user sending the email.
        body_html_base64 (str): The HTML body of the email, base64 encoded.
        tool_context (ToolContext): The tool context object.

    Returns:
        dict: A dictionary containing the status and message from the email service.
    """
    logging.info(f"Sending email to {to_address} with subject '{subject}'.")
    headers = {'Content-Type': 'application/json'}
    data = {
        'to_address': to_address,
        'subject': subject,
        'user_name': user_name,
        'body_html_base64': body_html_base64
    }
    url = "https://email-agent-backend-912533822336.us-central1.run.app/send-email"

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending email to {to_address}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

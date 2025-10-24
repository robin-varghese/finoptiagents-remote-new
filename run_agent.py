import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from app.agent import root_agent
from google.genai import types as genai_types


async def main():
    """Runs the agent with a sample query to send an email."""
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="app", user_id="test_user", session_id="test_session"
    )
    runner = Runner(
        agent=root_agent, app_name="app", session_service=session_service
    )
    
    import base64

    html_body = "<h1>Hello from FinOpti Agent!</h1><p>This is a test email.</p>"
    body_html_base64 = base64.b64encode(html_body.encode('utf-8')).decode('utf-8')

    # Example query to trigger the send_email tool
    query = f"Please send an email. The recipient is robinkv@gmail.com, the subject is 'Test Email from FinOpti Agent', the sender's name is 'FinOpti Agent', and the base64 encoded HTML body is '{body_html_base64}'."
    
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=genai_types.Content(
            role="user", 
            parts=[genai_types.Part.from_text(text=query)]
        ),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())

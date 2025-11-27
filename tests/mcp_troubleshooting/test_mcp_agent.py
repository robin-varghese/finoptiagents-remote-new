import nest_asyncio
nest_asyncio.apply()

import asyncio
import logging
from app.agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from dotenv import load_dotenv
import app.config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

import os
print(f"DEBUG: GOOGLE_API_KEY set? {'Yes' if os.environ.get('GOOGLE_API_KEY') else 'No'}")
print(f"DEBUG: GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
print(f"DEBUG: app.config.GOOGLE_PROJECT_ID: {app.config.GOOGLE_PROJECT_ID}")
print(f"DEBUG: app.config.GOOGLE_ZONE: {app.config.GOOGLE_ZONE}")

async def test_agent():
    print("--- Starting Agent Test ---")
    
    # Create a user message that requires using an MCP tool
    user_message = "List all VM instances in project 'vector-search-poc' and zone 'us-central1-a'."
    print(f"User Message: {user_message}")
    
    # Setup Runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="test_app",
        session_service=session_service
    )
    
    # Create session
    await session_service.create_session(
        app_name="test_app",
        user_id="test-user",
        session_id="test-session"
    )
    
    # Run the agent
    response_text = ""
    async for event in runner.run_async(
        user_id="test-user",
        session_id="test-session",
        new_message=Content(role="user", parts=[Part(text=user_message)])
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            response_text += event.content.parts[0].text
            
    print("--- Agent Response ---")
    print(response_text)
    print("--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_agent())

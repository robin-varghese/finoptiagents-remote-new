import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from app.utils.logging_config import setup_logging
from app.agent import finops_app
from google.genai import types as genai_types

# Configure logging immediately
setup_logging()


async def main():
    """Runs the agent with a sample query about VM deletions."""
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="finoptiagents_app", user_id="test_user", session_id="test_session"
    )
    runner = Runner(
        app=finops_app, session_service=session_service
    )
    
    query = "Use the run_bq_query tool to answer this: As part of the Budgeted & actual cost spent analysis, can you identify which are the good projects and which are the bad projects"

    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=genai_types.Content(
            role="user", 
            parts=[genai_types.Part.from_text(text=query)]
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())

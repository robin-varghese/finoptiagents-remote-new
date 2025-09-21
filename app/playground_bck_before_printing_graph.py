import asyncio
import streamlit as st
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import your root agent
from app.agent import root_agent

def get_or_create_eventloop():
    """Gets the running event loop or creates a new one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:  # 'get_running_loop' raises a RuntimeError if there is no running event loop
        # If there is no running loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="FinOps Agent Playground",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 FinOps Agent Playground")
st.write(
    "Interact with your agent locally. The agent has access to the tools you've defined."
)

# --- Session Management ---
# Initialize session state for chat history and session service
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_service" not in st.session_state:
    st.session_state.session_service = InMemorySessionService()
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session-1"
    # Create the session in the service
    loop = get_or_create_eventloop()
    loop.run_until_complete(
         st.session_state.session_service.create_session(
            app_name="finoptiagents-remote-new",
            user_id="streamlit-user",
            session_id=st.session_state.session_id,
        )
    )

# --- Agent Runner Initialization ---
# This runner will execute your agent's logic
runner = Runner(
    agent=root_agent,
    app_name="finoptiagents-remote-new",
    session_service=st.session_state.session_service,
)

# --- Chat History Display ---
# Display previous messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Agent Interaction ---
if prompt := st.chat_input("What would you like to do?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the message for the agent
    user_message = Content(role="user", parts=[Part.from_text(text=prompt)])

    # --- Run the Agent and Stream the Response ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # This async function will stream the response and return the complete text.
            async def stream_agent_response():
                response_accumulator = ""
                async for event in runner.run_async(
                    user_id="streamlit-user",
                    session_id=st.session_state.session_id,
                    new_message=user_message,
                ):
                    # Display intermediate tool calls for debugging
                    if event.get_function_calls():
                        func_call = event.get_function_calls()[0]
                        st.code(f"🛠️ Calling: {func_call.name}({func_call.args})", language="bash")

                    # Append text chunks to the full response
                    if event.content and event.content.parts:
                        text_chunk = event.content.parts[0].text
                        if text_chunk:
                            response_accumulator += text_chunk
                            message_placeholder.markdown(response_accumulator + "▌")

                message_placeholder.markdown(response_accumulator)
                return response_accumulator

            # Run the async function in the event loop and capture the full response
            loop = get_or_create_eventloop()
            full_response = loop.run_until_complete(stream_agent_response())

        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = f"Sorry, an error occurred: {e}"

    # Add the final assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
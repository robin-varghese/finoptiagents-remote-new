import asyncio
import streamlit as st
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from dotenv import load_dotenv
import json
import plotly.graph_objects as go
import re
import logging

# --- Custom CSS for Chat Bubbles ---
st.markdown("""
    <style>
        /* General styling for all chat messages */
        [data-testid="stChatMessage"] {
            border-radius: 20px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        /* ... (rest of your CSS is correct) ... */
        [data-testid="stChatMessage"]:has([data-testid="chat-message-container-user"]) {
            background-color: #E1F5FE;
        }
        [data-testid="stChatMessage"]:has([data-testid="chat-message-container-user"]) p {
            color: #01579B;
        }
        [data-testid="stChatMessage"]:has([data-testid="chat-message-container-assistant"]) {
            background-color: #F5F5F5;
        }
        [data-testid="stChatMessage"]:has([data-testid="chat-message-container-assistant"]) p {
            color: #212121;
        }
        .st-emotion-cache-1c7y2kd {
            background-color: rgba(100,100,100,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Import agents
from app.agent import root_agent, greeting_agent

# --- Sample Prompts ---
sample_prompts = {
    "FinOps Analyst / Cloud Financial Manager": [
        "What Data Analysis & Reporting can you do",
        "As part of the Budgeted & actual cost spent analysis, can you identify which are the good projects and which are the bad projects",
        "as part of Non-Compliance Analysis can you list out all projects in the category and the reasons and actions needed",
        "can you run utlization analysis and identify which are the projects spending more in production than lower environment",
        "can you run optimisation analysis and suggest which project has optimisation chances and why so",
        "can you suggest which cloud resources are good candidates for cost optimization",
        "tell me which project in our organization is using these resources and the resource utlization is less than 50%",
        "which are the projects having lower environments but doesn\'t have any tickets for release (change request or defects)",
        "can you generate a graph to send to my manager to show the cloud spend for every project",
    ],
    "Engineering Manager / Team Lead": [
        "who are the stakeholders of these projects",
        "who are managing the good projects in-terms of compliance.",
        "can you list all the GCP projects running in the project vector-search-poc in zone us-centrall-a",
        "can you check the CPU utilisation for this virtual machine",
        "can you delete the under utilised CPU from cloud. project name : vector-search-poc zone:us-centrall-a",
    ],
    "Compliance / Audit Team": [
        "so how many vms were deleted today",
        "can you do the query and tell me how many virtual machines were deleted yesterday",
        "who deleted the last VM",
        "how many vms were deleted by Robin",
        "can you check whether Robin has deleted any VMs",
        "can you check whether Robin Varghese has deleted any VMs",
        "when did Robin Varghese delete the VMs",
    ],
}

def get_or_create_eventloop():
    """Gets the running event loop or creates a new one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="FinOps Agent Playground", page_icon="🤖", layout="wide")
st.title("🤖 FinOps Agent Playground")
st.write("Interact with your agent locally. The agent has access to the tools you've defined.")

# --- Clicked Prompts Initialization ---
if "clicked_prompts" not in st.session_state:
    st.session_state.clicked_prompts = set()

# --- Sidebar with Sample Prompts ---
with st.sidebar:
    st.title("Sample Prompts")
    st.write("Click a prompt to use it.")
    for persona, prompts in sample_prompts.items():
        with st.expander(persona):
            for prompt_text in prompts:
                if prompt_text in st.session_state.clicked_prompts:
                    label = f"✅ {prompt_text}"
                else:
                    label = prompt_text
                if st.button(label):
                    st.session_state.prompt_from_sidebar = prompt_text
                    st.session_state.clicked_prompts.add(prompt_text)

# --- Session Management & Automatic Greeting ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.spinner("FinOpti is waking up..."):
        try:
            greeting_session_service = InMemorySessionService()
            greeting_runner = Runner(agent=greeting_agent, app_name="greeting_app", session_service=greeting_session_service)

            async def get_greeting():
                """Runs the greeting agent and captures its response."""
                await greeting_session_service.create_session(
                    app_name="greeting_app",
                    user_id="streamlit-user",
                    session_id="greeting-session"
                )
                response_text = ""
                initial_message = Content(role="user", parts=[Part(text="greet me")])
                async for event in greeting_runner.run_async(
                    user_id="streamlit-user",
                    session_id="greeting-session",
                    new_message=initial_message,
                ):
                    if event.content and event.content.parts and event.content.parts[0].text:
                        response_text += event.content.parts[0].text
                return response_text

            loop = get_or_create_eventloop()
            greeting_message = loop.run_until_complete(get_greeting())
            st.session_state.messages.append({"role": "assistant", "content": greeting_message})
        except Exception as e:
            logging.error(f"Failed to get initial greeting: {e}", exc_info=True)
            st.session_state.messages.append({"role": "assistant", "content": "Hello! I seem to be having trouble starting up. You can still ask me questions."})

if "session_service" not in st.session_state:
    st.session_state.session_service = InMemorySessionService()
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session-1"
    loop = get_or_create_eventloop()
    loop.run_until_complete(
         st.session_state.session_service.create_session(
            app_name="finoptiagents-remote-new",
            user_id="streamlit-user",
            session_id=st.session_state.session_id,
        )
    )

# --- Agent Runner Initialization ---
runner = Runner(
    agent=root_agent,
    app_name="finoptiagents-remote-new",
    session_service=st.session_state.session_service,
)

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], go.Figure):
            st.plotly_chart(message["content"])
        else:
            st.markdown(str(message["content"]))

# --- User Input and Agent Interaction ---
prompt = st.chat_input("What would you like to do?")
if "prompt_from_sidebar" in st.session_state and st.session_state.prompt_from_sidebar:
    prompt = st.session_state.prompt_from_sidebar
    st.session_state.prompt_from_sidebar = None

if prompt:
    logging.info(f"User input received: '{prompt}'")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    user_message = Content(role="user", parts=[Part(text=prompt)])

    # In app/playground.py

    with st.chat_message("assistant"):
        final_content_to_store = None 
        
        # --- THIS IS THE FIX: ADD THE FULL TRY...EXCEPT BLOCK ---
        try:
            thinking_placeholder = st.empty()
            
            async def run_agent_and_get_final_response(message_to_agent):
                response_text = ""
                thinking_steps = []
                async for event in runner.run_async(
                    user_id="streamlit-user",
                    session_id=st.session_state.session_id,
                    new_message=message_to_agent,
                ):
                    if hasattr(event, 'function_calls') and event.function_calls:
                        # Assuming single tool call for simplicity in display
                        func_call = event.function_calls[0]
                        thinking_steps.append(f"```bash\n🛠️ Calling Tool: {func_call.name}({func_call.args})\n```")
                        thinking_placeholder.markdown("\n".join(thinking_steps))

                    if hasattr(event, 'content') and event.content and event.content.role == 'model':
                        if event.content.parts and event.content.parts[0].text:
                            response_text += event.content.parts[0].text
                
                return response_text

            loop = get_or_create_eventloop()
            # In playground.py
            final_response = loop.run_until_complete(run_agent_and_get_final_response(user_message))
            logging.info(f"Agent returned final response.")
            thinking_placeholder.empty()
            
            # --- START: FINAL, CORRECTED RENDERING LOGIC ---
            final_content_to_store = final_response
            is_chart_rendered = False
            
            try:
                # First, try to load the entire response as JSON.
                parsed_json = json.loads(final_response)
                
                if isinstance(parsed_json, dict) and "data" in parsed_json and "layout" in parsed_json:
                    fig = go.Figure(parsed_json)
                    st.plotly_chart(fig)
                    final_content_to_store = fig
                    is_chart_rendered = True
            
            except json.JSONDecodeError:
                # Fallback to searching for a JSON block within the text.
                json_match = re.search(r'\{.*\}', final_response, re.DOTALL)
                if json_match:
                    chart_json_str = json_match.group(0)
                    try:
                        parsed_json = json.loads(chart_json_str)
                        if isinstance(parsed_json, dict) and "data" in parsed_json and "layout" in parsed_json:
                            intro_text = final_response[:json_match.start()].strip()
                            if intro_text:
                                st.markdown(intro_text)
                            
                            fig = go.Figure(parsed_json)
                            st.plotly_chart(fig)
                            final_content_to_store = fig
                            is_chart_rendered = True
                    except json.JSONDecodeError:
                        pass

            if not is_chart_rendered:
                st.markdown(final_response)
            
            logging.info("Final response rendered to UI.")
            # --- END: FINAL, CORRECTED RENDERING LOGIC ---

        # This `except` block corresponds to the main `try` at the top
        except Exception as e:
            logging.error(f"An error occurred during agent execution: {e}", exc_info=True)
            st.error(f"An error occurred: {e}")
            final_content_to_store = f"Sorry, an error occurred: {e}"
        # --- END OF FIX ---
    
    st.session_state.messages.append({"role": "assistant", "content": final_content_to_store})
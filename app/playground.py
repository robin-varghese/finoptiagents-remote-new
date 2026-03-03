import asyncio
import streamlit as st
import subprocess
from google.adk.runners import Runner
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from google.genai import errors as google_genai_errors
from dotenv import load_dotenv
import json
import plotly.graph_objects as go
import re
import logging
import traceback
import nest_asyncio
import warnings

# Suppress the AsyncClient.aclose() warning from google.genai
# This is a known issue with the ADK's internal async client management
warnings.filterwarnings("ignore", message="coroutine 'AsyncClient.aclose' was never awaited")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*AsyncClient.aclose.*")

from app.utils.logging_config import setup_logging
from app import config

# Configure logging using centralized strategy
setup_logging()

# Enable logging for tool modules explicitly if needed (optional as root logger covers them)
logging.getLogger('mcp_server.tools').setLevel(logging.INFO)
logging.getLogger('app.agent').setLevel(logging.INFO)
logging.getLogger('google.adk').setLevel(logging.INFO)

nest_asyncio.apply()

# Verify logging is working
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("🚀 FinOptiAgents Playground Starting...")
logger.info("=" * 80)

def get_gcloud_user():
    """Fetches the currently logged-in gcloud user email."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            capture_output=True, text=True, check=True
        )
        email = result.stdout.strip()
        if email:
            return email
    except Exception as e:
        logger.warning(f"Could not retrieve gcloud user: {e}")
    return "streamlit-user" # Fallback

CURRENT_USER = get_gcloud_user()
logger.info(f"Session User ID set to: {CURRENT_USER}")

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
        /* Smaller font for sidebar */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] button, [data-testid="stSidebar"] div {
            font-size: 0.85rem !important;
        }
        /* Loading Animation */
        .loading-dots {
            display: inline-block;
            position: relative;
            width: 64px;
            height: 24px;
        }
        .loading-dots div {
            position: absolute;
            top: 6px;
            width: 11px;
            height: 11px;
            border-radius: 50%;
            background: #2196F3;
            animation-timing-function: cubic-bezier(0, 1, 1, 0);
        }
        .loading-dots div:nth-child(1) {
            left: 6px;
            animation: loading-dots1 0.6s infinite;
        }
        .loading-dots div:nth-child(2) {
            left: 6px;
            animation: loading-dots2 0.6s infinite;
        }
        .loading-dots div:nth-child(3) {
            left: 26px;
            animation: loading-dots2 0.6s infinite;
        }
        .loading-dots div:nth-child(4) {
            left: 45px;
            animation: loading-dots3 0.6s infinite;
        }
        @keyframes loading-dots1 {
            0% { transform: scale(0); }
            100% { transform: scale(1); }
        }
        @keyframes loading-dots3 {
            0% { transform: scale(1); }
            100% { transform: scale(0); }
        }
        @keyframes loading-dots2 {
            0% { transform: translate(0, 0); }
            100% { transform: translate(19px, 0); }
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Import agents
from app.agent import finops_app, root_agent, greeting_agent

# --- Sample Prompts ---
sample_prompts = {
    "FinOps Analyst / Cloud Financial Manager": [
        "List all 16+ Google Cloud Recommenders",
        "What Data Analysis & Reporting can you do",
        "Send me an email about the information (explanatory) about FinOptiagents Data Analysis & Reporting capabilities? My name=Robin Varghese, email=robinkv@gmail.com, subject=FinOptiagents Data Analysis & Reporting capabilities",
        "As part of the Budgeted & actual cost spent analysis, can you identify which are the good projects and which are the bad projects",
        "as part of Non-Compliance Analysis can you list out all projects in the category and the reasons and actions needed",
        "Run utlization analysis and identify which are the projects spending more in lower environment than production environment",
        "Run optimisation analysis and suggest which project has optimisation chances and why so",
        "Suggest which cloud resources are good candidates for cost optimization",
        "Which project in our organization is using these resources and the resource utlization is less than 50%",
        "Which are the projects having lower environments but doesn\'t have any tickets for release (change request or defects)",
        "Generate a graph (Bar chart) to send to my manager to show the average cloud spend for every project per month. I would like to budget the cloud cost for next year based on this data",
        "Using a Pie chart show the cloud spend accross different projects per month in different environments. I would like to know the percentage of spend in each environment",
        "Using a line chart show the cloud spend accross different projects per month. I would like to know the trend in each month and find out whether any spikes are there in any month",
        "Review design document for project Alpha by comparing it with gs://finoptiagent-earb-designdocument2/Google Cloud Well-Architected Framework  _  Cloud Architecture Center  _  Google Cloud Documentation.pdf (in design document corpus) and let me know the gaps in the design document, especially regarding finops",
        "Review the design document with EARB for the projects and check which are projects deviated during the implementation in-terms of Cloud resources",
    ],
    "Engineering Manager / Team Lead": [
        "Who are the stakeholders of these projects",
        "Who are managing the good projects in-terms of compliance",
        "List all the GCP VMs/CPUs running in the project vector-search-poc in zone us-central1-a",
        "Check the CPU utilisation for this virtual machine",
        "Delete the under utilised CPU from cloud. project name : vector-search-poc zone:us-central1-a",
    ],
    "Compliance / Audit Team": [
        "How many vms were deleted today",
        "Query and tell me how many virtual machines were deleted yesterday",
        "Who deleted the last VM",
        "How many vms were deleted by Robin",
        "Check whether Robin has deleted any VMs",
        "Check whether Robin Varghese has deleted any VMs",
        "When did Robin Varghese delete the VMs",
    ],
    "Admin": [
        "list all the finops optimisation recommandations in gcp project vector-search-poc",
        "Create a new Vertex AI RAG corpus with the specified name finoptiagents_design_docs_rag using the docs in Google Cloud Storage gs://finoptiagent-earb-designdocument2",
        "Delete the Vertex AI RAG corpus with the specified name finoptiagents_design_docs_rag",
        "create an address named finoptiagents-teststatic-ip in region europe-west2",
        "delete the address named finoptiagents-teststatic-ip in region europe-west2",
    ]
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
                    user_id=CURRENT_USER,
                    session_id="greeting-session"
                )
                response_text = ""
                initial_message = Content(role="user", parts=[Part(text="greet me")])
                async for event in greeting_runner.run_async(
                    user_id=CURRENT_USER,
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
            app_name="finoptiagents_app",
            user_id=CURRENT_USER,
            session_id=st.session_state.session_id,
        )
    )

# --- Agent Runner Initialization (MOVED LOCAL FOR PLUGIN FLUSHING) ---
# runner = Runner(
#     app=finops_app,
#     session_service=st.session_state.session_service,
# )

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
            
            # SHOW LOADING ANIMATION
            thinking_placeholder.markdown("""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div class="loading-dots"><div></div><div></div><div></div><div></div></div>
                    <span style="color: #666; font-style: italic;">Thinking...</span>
                </div>
            """, unsafe_allow_html=True)
            
            async def run_agent_and_get_final_response(message_to_agent):
                response_text = ""
                thinking_steps = []
                
                # --- LOCAL SETUP FOR PLUGIN LIFECYCLE MANAGEMENT ---
                analytics_plugin = BigQueryAgentAnalyticsPlugin(
                    project_id=config.GOOGLE_PROJECT_ID,
                    dataset_id=config.BIGQUERY_DATASET_ID,
                    table_id=config.BIGQUERYAGENTANALYTICSPLUGIN_TABLE_ID,
                    location=config.GOOGLE_ZONE
                )
                
                # Recreate app instance locally to attach our local plugin instance
                # We reuse root_agent and Reflect plugin from global if needed, or just new list
                from app.agent import ReflectAndRetryToolPlugin # Import relevant potential plugins
                
                local_app = App(
                    name="finoptiagents_app",
                    root_agent=root_agent,
                    plugins=[
                        ReflectAndRetryToolPlugin(max_retries=3),
                        analytics_plugin
                    ]
                )
                
                local_runner = Runner(
                    app=local_app,
                    session_service=st.session_state.session_service,
                )
                
                try:
                    async for event in local_runner.run_async(
                        user_id=CURRENT_USER,
                        session_id=st.session_state.session_id,
                        new_message=message_to_agent,
                    ):
                        # DEBUG LOGGING START
                        logging.info(f"Runner Event received: {type(event)}")
                        if hasattr(event, 'content') and event.content:
                            logging.info(f"Event Content Role: {event.content.role}")
                            logging.info(f"Event Content Parts: {event.content.parts}")
                        if hasattr(event, 'function_calls') and event.function_calls:
                            logging.info(f"Event Function Calls: {event.function_calls}")
                    # DEBUG LOGGING END
                    
                    if hasattr(event, 'error_message') and event.error_message:
                        st.error(f"Agent Error: {event.error_message} (Code: {getattr(event, 'error_code', 'N/A')})")
                        logging.error(f"Agent Error Event: {event.error_message}")

                    if hasattr(event, 'function_calls') and event.function_calls:
                        # Assuming single tool call for simplicity in display
                        func_call = event.function_calls[0]
                        thinking_steps.append(f"```bash\n🛠️ Calling Tool: {func_call.name}({func_call.args})\n```")
                        thinking_placeholder.markdown("\n".join(thinking_steps))

                    if hasattr(event, 'content') and event.content and event.content.role == 'model':
                        if event.content.parts and event.content.parts[0].text:
                            response_text += event.content.parts[0].text
                finally:
                    logging.info("Closing analytics plugin (flushing logs)...")
                    if 'analytics_plugin' in locals() and analytics_plugin:
                        await analytics_plugin.close()

                return response_text

            # Use asyncio.run() for robust event loop management
            # If a loop is already running (e.g. Streamlit internal loop), asyncio.run will fail.
            # In that case, we fall back to creating a task or running in the existing loop.
            try:
                final_response = asyncio.run(run_agent_and_get_final_response(user_message))
            except RuntimeError as e:
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    # Fallback: We are already in a loop.
                    # Since we removed nest_asyncio, we can't use loop.run_until_complete re-entrantly.
                    # But we can try to await it if we were async... but we are not.
                    # We must use a new thread to run a fresh loop if we are blocked.
                    # OR, we can try to get the current loop and use it? No, run_until_complete fails if running.
                    # The only safe way in a running loop (without nest_asyncio) is to spawn a thread.
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, run_agent_and_get_final_response(user_message))
                        final_response = future.result()
                else:
                    raise e
            
            logging.info(f"Agent returned final response. Length: {len(final_response)}")
            logging.info(f"Final Response Content: {final_response!r}")
            thinking_placeholder.empty()
            
            # --- START: FINAL, CORRECTED RENDERING LOGIC ---
            final_content_to_store = final_response
            is_chart_rendered = False
            
            try:
                # 1. First, try to load the entire response as JSON.
                parsed_json = json.loads(final_response)
                
                # --- NEW: Enhanced Rendering for FinOps Schemas ---
                if isinstance(parsed_json, dict):
                    # Check for Chart JSON (Plotly)
                    if "data" in parsed_json and "layout" in parsed_json:
                        fig = go.Figure(parsed_json)
                        st.plotly_chart(fig)
                        final_content_to_store = fig
                        is_chart_rendered = True
                    
                    # Check for Budget Variance results
                    elif "projects_at_risk" in parsed_json:
                        st.subheader("📊 Budget Variance Analysis")
                        st.write(f"**Severity:** {parsed_json.get('severity', 'N/A').upper()} | **Total Analyzed:** {parsed_json.get('total_projects_analyzed', 'N/A')}")
                        import pandas as pd
                        df = pd.DataFrame(parsed_json["projects_at_risk"])
                        st.table(df)
                        is_chart_rendered = True
                    
                    # Check for Compliance results
                    elif "non_compliant_projects" in parsed_json:
                        st.subheader("🛡️ Compliance Audit")
                        st.write(f"**Severity:** {parsed_json.get('severity', 'N/A').upper()}")
                        st.error(f"Found {len(parsed_json['non_compliant_projects'])} non-compliant projects.")
                        st.write("**Issues identified:**")
                        for action in parsed_json.get("recommended_actions", []):
                            st.info(action)
                        is_chart_rendered = True

                    # Check for Utilization results
                    elif "anomalous_environments" in parsed_json or "low_utilization_resources" in parsed_json:
                        st.subheader("📈 Utilization Insights")
                        if "anomalous_environments" in parsed_json and parsed_json["anomalous_environments"]:
                            st.write("**Anomalous Environments (Lower > Prod Cost):**")
                            st.table(parsed_json["anomalous_environments"])
                        if "low_utilization_resources" in parsed_json and parsed_json["low_utilization_resources"]:
                            st.write("**Low Utilization Resources (<50%):**")
                            st.table(parsed_json["low_utilization_resources"])
                        st.success(f"Estimated Potential Savings: ${parsed_json.get('total_potential_savings', 0):,.2f}")
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
                        elif isinstance(parsed_json, dict) and "projects_at_risk" in parsed_json:
                            intro_text = final_response[:json_match.start()].strip()
                            if intro_text: st.markdown(intro_text)
                            st.table(parsed_json["projects_at_risk"])
                            is_chart_rendered = True
                    except json.JSONDecodeError:
                        pass


            if not is_chart_rendered:
                st.markdown(final_response)
            
            logging.info("Final response rendered to UI.")
            # --- END: FINAL, CORRECTED RENDERING LOGIC ---

        # This `except` block corresponds to the main `try` at the top
        except google_genai_errors.ServerError as e:
            logging.error(f"A server error occurred during agent execution: {e}", exc_info=True)
            error_message = "The model is currently unavailable or overloaded (503 Error). Please try again in a few moments."
            st.error(error_message)
            final_content_to_store = error_message
        except Exception as e:
            logging.error(f"An error occurred during agent execution: {e}", exc_info=True)
            st.error(f"An error occurred: {e}\n\nTraceback:\n```\n{traceback.format_exc()}\n```")
            final_content_to_store = f"Sorry, an error occurred: {e}"
        # --- END OF FIX ---
    
    st.session_state.messages.append({"role": "assistant", "content": final_content_to_store})
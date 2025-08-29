import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import asyncio

# Gemini API Key (move to env or secrets for production)
gemini_api_key =  "AIzaSyC673_pWB6IQ_SFwU0OzFc14NsFJNPNfXQ"

# Check API key
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

# Client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="Assistant",
    instructions=(
        "You are a helpful assistant. Always respond based on the conversation context. "
        "If unsure, politely say 'I'm not sure about that.' Keep responses concise and relevant."
    ),
    model=model
)

async def get_agent_response(user_input, chat_history):
    context = ""
    for chat in chat_history:
        sender = "User" if chat["sender"] == "user" else "Assistant"
        context += f"{sender}: {chat['message']}\n"
    context += f"User: {user_input}\nAssistant:"

    result = await Runner.run(agent, context, run_config=config)
    return result.final_output.strip()

def run_asyncio_task(task):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.ensure_future(task)
    else:
        return asyncio.run(task)

st.title("AI Chat Assistant ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat history
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(f"**🧑‍💻 You:** {chat['message']}")
    else:
        st.markdown(f"**🤖 Assistant:** {chat['message']}")

# Form style input with arrow button
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Type your message:")
    submit_button = st.form_submit_button("➤")  # Arrow button

    if submit_button and user_input.strip():
        st.session_state.chat_history.append({"sender": "user", "message": user_input})
        with st.spinner("Assistant is thinking... 💭"):
            response = run_asyncio_task(get_agent_response(user_input, st.session_state.chat_history))
            if asyncio.isfuture(response):
                response = asyncio.run(response)
        st.session_state.chat_history.append({"sender": "assistant", "message": response})
        st.experimental_rerun()
    elif submit_button:
        st.warning("Please type something before sending ")

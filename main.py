import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Streamlit page config
st.set_page_config(page_title="OpenRouter Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 OpenRouter Chatbot (LangChain + Streamlit)")
st.caption("Free model chatbot using OpenRouter API")


# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = None

# ✅ Production (Streamlit cloud) try secrets
try:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
except FileNotFoundError:
    pass

# ✅ Local fallback (.env)
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found. Add it in .env or Streamlit Secrets.")
    st.stop()


# Initialize LLM
# llm = ChatOpenAI(
#     model="openai/gpt-oss-120b:free",
#     api_key=OPENROUTER_API_KEY,
#     base_url="https://openrouter.ai/api/v1",
# )

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
)


# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert session messages into LangChain format
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(chat_history)
            bot_reply = response.content
            st.markdown(bot_reply)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

st.set_page_config(page_title="Agentic Chatbot", page_icon="💬", layout="centered")


# ---------------------------------------------------------------
# Graph (built once, then reused on every Streamlit rerun)
# ---------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@st.cache_resource
def build_chatbot():
    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        timeout=60,
        max_retries=5,
        max_tokens=4096,
        reasoning_effort="low",
    )

    def chat_node(state: ChatState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    # MemorySaver lives in RAM. Cached here so it survives reruns.
    return graph.compile(checkpointer=MemorySaver())


chatbot = build_chatbot()

# One thread_id per browser session = one separate conversation
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def get_history() -> list:
    """Read the conversation back from the checkpointer."""
    state = chatbot.get_state(config)
    if state and state.values:
        return state.values.get("messages", [])
    return []


def token_stream(user_text: str):
    """Yield the reply piece by piece so it types out on screen."""
    for chunk, meta in chatbot.stream(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
        stream_mode="messages",
    ):
        if (
            meta.get("langgraph_node") == "chat_node"
            and isinstance(chunk.content, str)
            and chunk.content
        ):
            yield chunk.content


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Chatbot")
    st.caption(f"Model: `{MODEL_NAME}`")

    if st.button("New chat", use_container_width=True):
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption("Conversation thread")
    st.code(st.session_state["thread_id"][:8], language=None)
    st.caption(
        "Memory is kept in RAM by LangGraph's MemorySaver. "
        "Refreshing the page or restarting the app starts a new conversation."
    )

# ---------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------
st.title("Agentic Chatbot")
st.caption("A LangGraph chatbot with conversation memory.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()

history = get_history()

if not history:
    st.info("Ask me anything to get started. I remember what we talk about in this chat.")

for msg in history:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.type == "ai" and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

prompt = st.chat_input("Type your message")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            st.write_stream(token_stream(prompt))
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in ("413", "429", "rate_limit")):
                st.error("Groq rate limit reached. Wait a minute and try again.")
            else:
                st.error(f"Something went wrong: {msg}")
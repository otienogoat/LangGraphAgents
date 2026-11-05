"""Backend for the complete Agentic Chatbot using LangGraph.

This module contains:

* the PDF knowledge base (RAG) used by the sidebar uploader,
* the LangGraph agent and its tools,
* SQLite checkpointing and conversation-title persistence,
* human-in-the-loop (HITL) approval before sensitive tools run.

The Streamlit frontend lives in ``app.py``. Run ``python backend.py`` for a
small terminal version of the same agent.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from openai import APIStatusError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# ---------------------------------------------------------------------------
# PDF KNOWLEDGE BASE (RAG)
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(BASE_DIR, 'faiss_db')
DOCS_DIR = os.path.join(BASE_DIR, 'documents')
MANIFEST_PATH = os.path.join(DB_PATH, 'manifest.json')

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_CHARS_PER_DOC = 1200
MAX_TOTAL_CHARS = 5000

_rag_lock = threading.Lock()
_embeddings: FastEmbedEmbeddings | None = None
_vector_store: FAISS | None = None
_retriever = None


def get_embeddings() -> FastEmbedEmbeddings:
    """Create the embedding model once per process.

    ``batch_size=16`` avoids the large ONNX attention buffer that the default
    batch size reserves on low-memory machines.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(
            model_name='BAAI/bge-small-en-v1.5',
            batch_size=16,
            max_length=512,
        )
    return _embeddings


def index_exists() -> bool:
    return os.path.isfile(os.path.join(DB_PATH, 'index.faiss'))


def index_stats() -> dict:
    """Cheap index metadata, read without loading the FAISS index."""
    try:
        with open(MANIFEST_PATH, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {} if not index_exists() else {'files': [], 'chunks': None}


def _reset_cache() -> None:
    global _vector_store, _retriever
    with _rag_lock:
        _vector_store = None
        _retriever = None


def ingest_rag_document(file_path, db_path: str = DB_PATH) -> dict:
    """Build a FAISS index from one PDF (or a list of PDFs) and save it."""
    paths = [file_path] if isinstance(file_path, (str, os.PathLike)) else list(file_path)
    paths = [os.path.abspath(os.fspath(path)) for path in paths]
    if not paths:
        raise ValueError('No PDF file given')

    documents = []
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        documents.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError('The PDF contained no extractable text')

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    os.makedirs(db_path, exist_ok=True)
    vector_store.save_local(db_path)

    stats = {
        'files': [os.path.basename(path) for path in paths],
        'pages': len(documents),
        'chunks': len(chunks),
        'vectors': int(vector_store.index.ntotal),
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(db_path, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(stats, fh, indent=1)

    _reset_cache()
    return stats


def get_retriever(db_path: str = DB_PATH):
    """Load ``faiss_db`` once and return a similarity retriever (or None)."""
    global _vector_store, _retriever

    if not os.path.isfile(os.path.join(db_path, 'index.faiss')):
        return None

    with _rag_lock:
        if _retriever is None:
            _vector_store = FAISS.load_local(
                db_path,
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            _retriever = _vector_store.as_retriever(
                search_type='similarity',
                search_kwargs={'k': TOP_K},
            )
        return _retriever


def clear_rag(remove_documents: bool = True) -> None:
    """Delete the index (and, by default, the uploaded PDFs)."""
    import shutil

    _reset_cache()
    shutil.rmtree(DB_PATH, ignore_errors=True)
    if remove_documents:
        shutil.rmtree(DOCS_DIR, ignore_errors=True)


def indexed_pdf_paths() -> list[str]:
    """Return every PDF in ``documents/`` (used to rebuild the index)."""
    if not os.path.isdir(DOCS_DIR):
        return []
    return sorted(
        os.path.join(DOCS_DIR, name)
        for name in os.listdir(DOCS_DIR)
        if name.lower().endswith('.pdf')
    )


# ---------------------------------------------------------------------------
# LLM + TOOLS
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model='openai/gpt-oss-120b',
    # Groq exposes an OpenAI-compatible API.
    openai_api_key=os.getenv('GROQ_API_KEY'),
    openai_api_base='https://api.groq.com/openai/v1',
    # Groq's free tier counts max_tokens towards the per-minute budget, so
    # keep the reply budget small enough for tool results + prompt.
    max_tokens=2048,
)

search_tool = TavilySearch(
    max_results=5,
    topic='general',
    search_depth='advanced',
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2+2, math.sqrt(16), 10*5
    """
    try:
        allowed = {
            'math': math,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
        }
        result = eval(expression, {'__builtins__': {}}, allowed)
        return str(result)
    except Exception as exc:
        return f'Calculation error: {exc}'


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a given ticker symbol."""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol.upper(),
        'apikey': os.getenv('ALPHA_VANTAGE_API_KEY', '167WH59FQD5QXYOA'),
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    NOTE: This is a mock implementation:
    - No real brokerage API is called.
    - It simply returns a confirmation payload.
    """
    return {
        "status": "success",
        "message": f"Purchase order placed for {quantity} shares of {symbol}.",
        "symbol": symbol,
        "quantity": quantity,
    }


@tool
def rag_tool(query: str) -> str:
    """Search the PDF documents in the knowledge base.

    Use this tool when the question is about the content of the PDF files the
    user uploaded (notes, papers, manuals, reports). Prefer it over web search
    whenever the user refers to "the PDF", "the notes" or "the document".

    Args:
        query: The question or search phrase used to retrieve PDF content.
    """
    retriever = get_retriever()
    if retriever is None:
        return (
            'No PDF has been indexed yet, so there is nothing to search. '
            'Tell the user to attach a PDF with the + button in the chat box '
            '(only PDFs are supported), then ask their question again.'
        )

    documents = retriever.invoke(query)
    if not documents:
        return 'No relevant information was found in the indexed PDFs.'

    formatted_documents = []
    total_chars = 0

    for index, document in enumerate(documents, start=1):
        remaining = MAX_TOTAL_CHARS - total_chars
        if remaining <= 0:
            break

        source = os.path.basename(document.metadata.get('source', 'Unknown source'))
        page = document.metadata.get(
            'page_label',
            document.metadata.get('page', 'Unknown'),
        )
        content = ' '.join(document.page_content.split())[:min(MAX_CHARS_PER_DOC, remaining)]

        entry = (
            f'Document {index}\n'
            f'Source: {source}\n'
            f'Page: {page}\n'
            f'Content: {content}'
        )
        formatted_documents.append(entry)
        total_chars += len(entry) + 2

    return '\n\n'.join(formatted_documents)


tools = [rag_tool, search_tool, calculator, get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = SystemMessage(content=(
    'You are a helpful Agentic Chatbot with access to several tools.\n\n'

    'Tool usage instructions:\n'
    '- Use `rag_tool` for questions about the uploaded PDF or document. '
    'Always retrieve relevant document content before answering PDF-related questions.\n'
    '- Use `tavily_search` for current events, recent information, or information '
    'that requires an internet search.\n'
    '- Use `calculator` for mathematical calculations. Do not calculate complex '
    'expressions manually when the calculator is available.\n'
    '- Use `get_stock_price` when the user asks for the current price of a stock.\n'
    '- Use `purchase_stock` when the user asks to buy shares. It is a mock '
    'paper-trading tool: no real brokerage API is called and no real order is '
    'placed. The app automatically pauses this tool call for human approval, so '
    'call the tool directly instead of asking the user to confirm first.\n\n'

    'Answer general questions directly when no tool is required. '
    'Do not invent information from the uploaded document. '
    'If the user asks about a PDF but no document is available, tell them to '
    'attach a PDF with the + button in the chat box (or add one under the '
    "sidebar's 'Knowledge base' section). "
    'Only PDFs are supported - never claim you can accept attachments (images, '
    'Word files, etc.) in the chat box itself. '
    'After receiving a tool result, provide a clear and helpful final answer that '
    'cites the page numbers when the answer comes from the PDF.'
))


def invoke_with_retry(runnable, messages, max_attempts=3, wait_seconds=65):
    """Wait out Groq's per-minute rate-limit window instead of crashing."""
    for attempt in range(1, max_attempts + 1):
        try:
            return runnable.invoke(messages)
        except APIStatusError as exc:
            if exc.status_code not in (413, 429) or attempt == max_attempts:
                raise
            print(
                f'[HTTP {exc.status_code}] rate limited; waiting {wait_seconds}s '
                f'(attempt {attempt}/{max_attempts})',
                flush=True,
            )
            time.sleep(wait_seconds)


def chat_node(state: ChatState):
    # The system message is prepended here (not stored in state) so the
    # checkpointed history stays clean across turns.
    messages = [SYSTEM_PROMPT, *state['messages']]
    response = invoke_with_retry(llm_with_tools, messages)
    return {'messages': [response]}


# ---------------------------------------------------------------------------
# HUMAN-IN-THE-LOOP (HITL)
# ---------------------------------------------------------------------------

APPROVAL_YES = {'yes', 'y', 'approve', 'approved', 'allow', 'ok', 'true'}

# Which tools must be approved before they run. Set HITL_TOOLS in .env to a
# comma-separated list of tool names, or "all"/"*" to gate every tool.
HITL_TOOLS = {
    name.strip()
    for name in os.getenv('HITL_TOOLS', 'purchase_stock').split(',')
    if name.strip()
}


def requires_approval(tool_name: str | None) -> bool:
    return (
        'all' in HITL_TOOLS
        or '*' in HITL_TOOLS
        or tool_name in HITL_TOOLS
    )


def _normalise_approval(decision):
    """Accept the shapes a frontend may resume the graph with."""
    if isinstance(decision, dict):
        raw = decision.get('approved', decision.get('decision', False))
        reason = str(decision.get('reason') or '').strip()
    else:
        raw = decision
        reason = ''

    if isinstance(raw, str):
        approved = raw.strip().lower() in APPROVAL_YES
    else:
        approved = bool(raw)
    return approved, reason


def _describe_tool_calls(tool_calls):
    """Trimmed, JSON-friendly description of the calls awaiting approval."""
    return [
        {
            'id': call.get('id'),
            'name': call.get('name'),
            'args': call.get('args') or {},
            'requires_approval': requires_approval(call.get('name')),
        }
        for call in tool_calls
    ]


def approval_node(state: ChatState):
    """Pause before a gated tool executes and ask a human to approve it.

    LangGraph replays this node from the beginning when the graph resumes, so
    ``interrupt`` returns the value supplied through ``Command(resume=...)``.
    """
    last_message = state['messages'][-1]
    tool_calls = getattr(last_message, 'tool_calls', None) or []
    gated_calls = [call for call in tool_calls if requires_approval(call.get('name'))]
    if not gated_calls:
        return {}  # nothing sensitive -> the edge routes straight to the tools

    decision = interrupt({
        'type': 'tool_approval',
        'question': next(
            (
                message.content
                for message in reversed(state['messages'])
                if isinstance(message, HumanMessage) and message.content
            ),
            '',
        ),
        'tool_calls': _describe_tool_calls(tool_calls),
        'instructions': (
            'Approve to run these tools, or reject to stop the action. '
            'Resume with {"approved": true/false, "reason": "..."}.'
        ),
    })

    approved, reason = _normalise_approval(decision)
    if approved:
        # Keep the AI tool-call message as the last message so tools_condition
        # routes to the tools node.
        return {}

    # Every pending tool call needs a ToolMessage even when rejected, otherwise
    # the history is invalid for the next LLM request.
    rejection = (
        'Tool call was rejected by the user.'
        if not reason
        else f'Tool call was rejected by the user: {reason}'
    )
    rejected_messages = [
        ToolMessage(
            content=rejection,
            tool_call_id=call.get('id'),
            name=call.get('name'),
        )
        for call in tool_calls
    ]
    tool_names = ', '.join(call.get('name') or 'tool' for call in tool_calls)
    final_message = AIMessage(
        content=(
            f'I did not run {tool_names} because the action was rejected.'
            + (f' Reason: {reason}' if reason else '')
        )
    )
    return {'messages': [*rejected_messages, final_message]}


# ---------------------------------------------------------------------------
# GRAPH + SQLITE PERSISTENCE
# ---------------------------------------------------------------------------

tool_node = ToolNode(tools)

DB_FILE = os.path.join(BASE_DIR, 'chatbot_conversations.db')
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('approval', approval_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
# If the LLM asked for a tool -> ask the human first; otherwise finish.
graph.add_conditional_edges(
    'chat_node',
    tools_condition,
    {'tools': 'approval', END: END},
)
# Approval leaves the AI tool-call message in place for the tools node, while a
# rejection appends a final AI message so tools_condition routes to END.
graph.add_conditional_edges('approval', tools_condition)
# After the tools run, go back to the LLM so it can use the results.
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)


def init_titles_table() -> None:
    """Make sure the database has a place to store conversation titles."""
    with conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS thread_titles ('
            'thread_id TEXT PRIMARY KEY, title TEXT NOT NULL)'
        )


def save_title(thread_id: str, title: str) -> None:
    """Persist a conversation title in the database."""
    init_titles_table()
    with conn:
        conn.execute(
            'INSERT INTO thread_titles (thread_id, title) VALUES (?, ?) '
            'ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title',
            (thread_id, title),
        )


def get_saved_title(thread_id: str) -> str | None:
    try:
        row = conn.execute(
            'SELECT title FROM thread_titles WHERE thread_id = ?',
            (thread_id,),
        ).fetchone()
    except Exception:
        return None
    return row[0] if row else None


def list_thread_ids() -> list[str]:
    """Return all conversation ids, oldest checkpoint first."""
    try:
        rows = conn.execute(
            'SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid'
        ).fetchall()
    except Exception:
        return []  # brand-new database: the checkpoints table doesn't exist yet
    return [row[0] for row in rows]


def delete_thread_data(thread_id: str) -> None:
    """Remove all persisted data of a conversation from the database."""
    with conn:
        for table in ('checkpoints', 'writes', 'thread_titles'):
            try:
                conn.execute(f'DELETE FROM {table} WHERE thread_id = ?', (thread_id,))
            except Exception:
                pass  # table might not exist yet


def load_thread_history(thread_id: str) -> list[dict]:
    """Return the visible user/assistant history of a conversation."""
    try:
        state = chatbot.get_state({'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', []) if state.values else []
    except Exception:
        return []

    history = []
    for message in messages:
        if message.type == 'human':
            role = 'user'
        elif message.type == 'ai':
            role = 'assistant'
        else:
            continue  # ignore tool/system messages
        content = message_text(message.content)
        if content:
            history.append({'role': role, 'content': content})
    return history


def get_pending_interrupt(thread_id: str):
    """Return the HITL request a paused conversation is waiting on, or None."""
    try:
        snapshot = chatbot.get_state({'configurable': {'thread_id': thread_id}})
    except Exception:
        return None
    return snapshot.interrupts[0].value if snapshot.interrupts else None


def thread_config(thread_id: str, thread_title: str | None = None) -> dict:
    """Shared LangGraph config for the Streamlit app and the CLI."""
    config = {
        'configurable': {'thread_id': thread_id},
        'metadata': {'thread_id': thread_id},
        'run_name': 'chat_trace',
    }
    if thread_title:
        config['metadata']['thread_title'] = thread_title
    return config


def message_text(content) -> str:
    """Flatten a LangChain message content (string or content blocks) to text."""
    if isinstance(content, list):
        return ''.join(
            part.get('text', '') if isinstance(part, dict) else str(part)
            for part in content
        )
    if isinstance(content, str):
        return content
    return str(content) if content is not None else ''


# ---------------------------------------------------------------------------
# TERMINAL CLI (optional; the main UI is app.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import uuid

    # Windows consoles often default to cp1252, which cannot print every
    # character the LLM emits; reconfigure to UTF-8 to avoid crashes.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    init_titles_table()
    thread_id = str(uuid.uuid4())
    CONFIG = thread_config(thread_id)

    def print_approval_request(request):
        print('\n⚠️  Human approval required')
        for call in request.get('tool_calls', []):
            if not call.get('requires_approval', True):
                continue
            args = json.dumps(call.get('args') or {}, ensure_ascii=False)
            print(f"   • {call.get('name')}({args})")

    print('Agentic Chat (backend CLI) - type /exit to quit')
    while True:
        user_input = input('You: ').strip()
        if not user_input:
            continue
        if user_input.lower() in ('/exit', '/quit'):
            break

        payload = {'messages': [HumanMessage(content=user_input)]}
        while True:
            print('Bot: ', end='', flush=True)
            for message_chunk, metadata in chatbot.stream(
                payload,
                config=CONFIG,
                stream_mode='messages',
            ):
                text = message_text(message_chunk.content)
                if message_chunk.type == 'tool':
                    if text:
                        print(
                            f'\n   ↳ {getattr(message_chunk, "name", "tool")}: '
                            f'{text[:500]}',
                            flush=True,
                        )
                    continue
                if message_chunk.type == 'ai' and text:
                    print(text, end='', flush=True)
            print()

            state = chatbot.get_state(CONFIG)
            if not state.interrupts:
                break

            request = state.interrupts[0].value
            print_approval_request(request)
            answer = input('Approve? [y/N] ').strip().lower()
            if answer in APPROVAL_YES:
                payload = Command(resume={'approved': True})
            else:
                reason = input('Reason (optional): ').strip()
                payload = Command(resume={'approved': False, 'reason': reason})

    conn.close()

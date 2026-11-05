"""Streamlit frontend for the complete Agentic Chatbot using LangGraph.

Run with:

    streamlit run app.py
"""

import json
import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from backend import (
    DOCS_DIR,
    chatbot,
    clear_rag,
    delete_thread_data,
    get_pending_interrupt,
    get_saved_title,
    index_exists,
    index_stats,
    indexed_pdf_paths,
    ingest_rag_document,
    init_titles_table,
    list_thread_ids,
    llm,
    load_thread_history,
    message_text,
    save_title,
    thread_config,
)


# ---------------------------------------------------------------------------
# CONVERSATION HELPERS
# ---------------------------------------------------------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Add a conversation id to the sidebar list if it is not there yet."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def new_chat():
    """Start a brand-new (empty) conversation and make it the active one."""
    new_thread_id = generate_thread_id()
    st.session_state['threads_messages'][new_thread_id] = []
    st.session_state['thread_titles'][new_thread_id] = 'New chat'
    add_thread(new_thread_id)
    st.session_state['current_thread'] = new_thread_id
    st.session_state['message_history'] = st.session_state['threads_messages'][new_thread_id]


def load_conversation(thread_id):
    """Open an existing conversation (used by the sidebar buttons)."""
    if thread_id in st.session_state['threads_messages']:
        st.session_state['current_thread'] = thread_id
        st.session_state['message_history'] = st.session_state['threads_messages'][thread_id]


def delete_thread(thread_id):
    """Remove a conversation from the sidebar and from the database."""
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    st.session_state['threads_messages'].pop(thread_id, None)
    st.session_state['thread_titles'].pop(thread_id, None)
    delete_thread_data(thread_id)

    if thread_id == st.session_state['current_thread']:
        if st.session_state['chat_threads']:
            load_conversation(st.session_state['chat_threads'][0])
        else:
            new_chat()


def load_saved_threads():
    """Restore conversations persisted by earlier sessions."""
    for thread_id in list_thread_ids():
        if thread_id in st.session_state['chat_threads']:
            continue

        history = load_thread_history(thread_id)
        if not history:
            continue  # never a real conversation -> don't show it

        add_thread(thread_id)
        st.session_state['threads_messages'][thread_id] = history
        st.session_state['thread_titles'][thread_id] = (
            get_saved_title(thread_id) or history[0]['content'][:40]
        )

    if st.session_state['chat_threads']:
        # Reopen the most recent conversation instead of a blank one.
        latest = st.session_state['chat_threads'][-1]
        st.session_state['current_thread'] = latest
        st.session_state['message_history'] = st.session_state['threads_messages'][latest]


def generate_title(thread_id):
    """Ask the LLM for a short title based on the first exchange."""
    history = st.session_state['threads_messages'].get(thread_id, [])
    transcript = '\n'.join(
        f"{m['role']}: {m['content']}" for m in history if m['content']
    )[:1200]
    try:
        response = llm.invoke([
            (
                'system',
                'Generate a very short conversation title (3-5 words) based on '
                'the conversation topic. Reply with the title text only - no '
                'quotes, no punctuation at the end.',
            ),
            ('user', transcript),
        ])
        title = response.content.strip().strip('"\'.').strip()
        if not title:
            raise ValueError('empty title')
    except Exception:
        title = history[0]['content'][:40] if history and history[0]['content'] else 'New chat'

    st.session_state['thread_titles'][thread_id] = title
    save_title(thread_id, title)


# ---------------------------------------------------------------------------
# KNOWLEDGE BASE (RAG)
# ---------------------------------------------------------------------------

def save_uploaded_pdfs(uploaded_files):
    """Store uploads in documents/ and return every PDF in that folder."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    for uploaded in uploaded_files:
        with open(os.path.join(DOCS_DIR, uploaded.name), 'wb') as fh:
            fh.write(uploaded.getbuffer())
    return indexed_pdf_paths()


# ---------------------------------------------------------------------------
# PAGE + STYLING
# ---------------------------------------------------------------------------

st.set_page_config(page_title='Agentic Chatbot', page_icon='💬', layout='wide')

st.markdown(
    """
    <style>
        /* dark sidebar */
        section[data-testid="stSidebar"] {
            background-color: #171717;
            border-right: 1px solid #2f2f2f;
        }
        /* all sidebar buttons become clean list rows */
        section[data-testid="stSidebar"] .stButton > button {
            background-color: transparent;
            color: #ececec;
            border: none;
            box-shadow: none;
            border-radius: 8px;
            width: 100%;
            justify-content: flex-start;
            padding: 0.5rem 0.75rem;
            font-size: 0.875rem;
            text-align: left;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: #2f2f2f;
            color: #ffffff;
        }
        section[data-testid="stSidebar"] .stButton > button p {
            font-weight: 500;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #2f2f2f;
        }
        /* remove extra padding under the sidebar header */
        section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
if 'threads_messages' not in st.session_state:
    st.session_state['threads_messages'] = {}
if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'current_thread' not in st.session_state:
    st.session_state['current_thread'] = None

init_titles_table()
load_saved_threads()

if st.session_state['current_thread'] is None:
    new_chat()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.5rem;padding:0.75rem 0.5rem;">'
        '<span style="font-size:1.35rem;">💬</span>'
        '<span style="font-size:1.05rem;font-weight:700;color:#ececec;">Agentic Chat</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.button('➕  New chat', on_click=new_chat, use_container_width=True)

    # ---------------------- PDF knowledge base ----------------------
    st.divider()
    st.markdown(
        '<div style="padding:0.6rem 0.75rem 0.25rem;font-size:0.75rem;'
        'text-transform:uppercase;letter-spacing:0.05em;color:#8e8e8e;">Knowledge base</div>',
        unsafe_allow_html=True,
    )

    stats = index_stats()
    if index_exists():
        files = ', '.join(stats.get('files') or []) or 'indexed PDF'
        st.caption(f'📚 {stats.get("chunks") or "?"} chunks · {files}')
    else:
        st.caption('📚 No PDF indexed yet — attach one with + below, or upload here.')

    # Bumping kb_version resets the uploader once files are indexed.
    upload_key = f'pdf_uploader_{st.session_state.get("kb_version", 0)}'
    uploaded = st.file_uploader(
        'Add PDF notes',
        type=['pdf'],
        accept_multiple_files=True,
        label_visibility='collapsed',
        key=upload_key,
    )

    if uploaded:
        signature = tuple(sorted((file.name, file.size) for file in uploaded))
        if st.session_state.get('kb_signature') != signature:
            with st.spinner('Reading and indexing PDFs…'):
                paths = save_uploaded_pdfs(uploaded)
                stats = ingest_rag_document(paths)
            st.session_state['kb_signature'] = signature
            st.session_state['kb_version'] = st.session_state.get('kb_version', 0) + 1
            st.rerun()

    col_reindex, col_clear = st.columns(2)

    if col_reindex.button('🔄 Rebuild', use_container_width=True, key='kb_rebuild'):
        paths = indexed_pdf_paths()
        if not paths:
            st.info('Upload a PDF first.')
        else:
            with st.spinner('Rebuilding the index…'):
                ingest_rag_document(paths)
            st.rerun()

    if col_clear.button('🗑 Clear', use_container_width=True, key='kb_clear'):
        clear_rag()
        st.session_state['kb_version'] = st.session_state.get('kb_version', 0) + 1
        st.session_state.pop('kb_signature', None)
        st.rerun()

    st.divider()
    st.markdown(
        '<div style="padding:0.6rem 0.75rem 0.25rem;font-size:0.75rem;'
        'text-transform:uppercase;letter-spacing:0.05em;color:#8e8e8e;">Conversations</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Newest conversation at the top, like ChatGPT.
    for thread_id in reversed(st.session_state['chat_threads']):
        title_row, delete_row = st.columns([7.5, 1])
        is_active = thread_id == st.session_state['current_thread']
        icon = '🟢' if is_active else '💬'
        with title_row:
            st.button(
                f'{icon} {st.session_state["thread_titles"][thread_id]}',
                key=f'thread_{thread_id}',
                on_click=load_conversation,
                args=(thread_id,),
                use_container_width=True,
            )
        with delete_row:
            st.button(
                '🗑',
                key=f'delete_{thread_id}',
                on_click=delete_thread,
                args=(thread_id,),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# CHAT RENDERING + TOOL PILLS
# ---------------------------------------------------------------------------

st.title('Agentic Chatbot with LangGraph')

TOOL_LABELS = {
    'tavily_search': ('🔎', 'Searched the web'),
    'web_search': ('🔎', 'Searched the web'),
    'rag_tool': ('📚', 'Searched the PDF notes'),
    'calculator': ('🧮', 'Ran a calculation'),
    'get_stock_price': ('📈', 'Checked the stock price'),
    'purchase_stock': ('💵', 'Placed a mock stock order'),
}


def tool_pill_label(name):
    icon, label = TOOL_LABELS.get(name, ('🛠️', f'Used {name}'))
    return f'{icon} {label}'


def tool_input_summary(name, args_json):
    """Short 'what it was called with' text for the tool pill."""
    try:
        args = json.loads(args_json or '{}')
    except Exception:
        return args_json or ''
    if name in ('tavily_search', 'web_search', 'rag_tool'):
        return args.get('query', '')
    if name == 'calculator':
        return args.get('expression', '')
    if name == 'get_stock_price':
        return args.get('symbol', '')
    if name == 'purchase_stock':
        return f"{args.get('symbol', '?')} × {args.get('quantity', '?')} shares"
    return ', '.join(f'{key}={value}' for key, value in args.items())


def result_preview(result, limit=600):
    """Trimmed raw result, shown inside the expanded pill."""
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + ' …'


def stream_assistant_reply(payload, config, preopened_calls=None):
    """Stream one graph run into a single assistant bubble.

    ``payload`` is either the initial user-message dict or a
    ``Command(resume=...)`` after a HITL approval. ``preopened_calls``
    recreates the tool pills for a run that was paused at an interrupt.
    """
    ai_message = ''
    response_box = None
    open_pills = {}          # tool_call_id -> pill currently running
    pills = []

    def make_pill(call_id, name, args_json=''):
        box = st.status(f'🛠️ {tool_pill_label(name or "tool")}…', expanded=True)
        pill = {
            'id': call_id,
            'name': name or 'tool',
            'args': args_json,
            'box': box,
            'done': False,
        }
        pills.append(pill)
        if call_id:
            open_pills[call_id] = pill
        return pill

    def finish_pill(pill, name, content=''):
        with pill['box']:
            st.caption(f'**{tool_input_summary(name, pill["args"]) or "—"}**')
            if content:
                st.text(result_preview(content))
        pill['box'].update(
            label=tool_pill_label(name),
            state='complete',
            expanded=False,
        )
        pill['done'] = True
        if pill.get('id'):
            open_pills.pop(pill['id'], None)

    def find_open_pill(call_id, name):
        if call_id and call_id in open_pills:
            return open_pills[call_id]
        return next(
            (pill for pill in pills if not pill['done'] and pill['name'] == name),
            None,
        )

    with st.chat_message('assistant'):
        # Rebuild the pills that existed when the graph paused, so the resumed
        # run shows the approved action's input and result.
        for call in preopened_calls or []:
            make_pill(
                call.get('id'),
                call.get('name'),
                json.dumps(call.get('args') or {}, default=str),
            )

        for message_chunk, metadata in chatbot.stream(
            payload,
            config=config,
            stream_mode='messages',
        ):
            text = message_text(message_chunk.content)
            tool_calls = getattr(message_chunk, 'tool_call_chunks', None) or []

            # A finished tool -> fold its result into the matching pill.
            if message_chunk.type == 'tool':
                name = getattr(message_chunk, 'name', None) or 'tool'
                call_id = getattr(message_chunk, 'tool_call_id', None)
                pill = find_open_pill(call_id, name)
                if pill is None:
                    pill = make_pill(call_id, name)
                finish_pill(pill, name, text)
                continue

            # The LLM is requesting a tool call -> open/update a pill.
            if tool_calls:
                call = tool_calls[0]
                call_id = call.get('id')
                name = call.get('name') or 'tool'
                pill = find_open_pill(call_id, name) if call_id else None
                if pill is None:
                    pill = make_pill(call_id, name)
                elif call_id and not pill.get('id'):
                    pill['id'] = call_id
                    open_pills[call_id] = pill
                if call.get('name'):
                    pill['name'] = call['name']
                    pill['box'].update(label=f'{tool_pill_label(call["name"])}…')
                if call.get('args'):
                    pill['args'] += call['args']
                if not text:
                    continue

            # Plain answer tokens -> stream them into the response box.
            if text:
                ai_message += text
                if response_box is None:
                    response_box = st.empty()
                response_box.markdown(ai_message)

        # Safety net: collapse anything still open (e.g. a tool that never
        # returned) so the UI never gets stuck on a spinner.
        for pill in pills:
            if not pill['done']:
                finish_pill(pill, pill['name'])

    return ai_message


# Loading the conversation history of the selected thread.
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


# ---------------------------------------------------------------------------
# HUMAN-IN-THE-LOOP APPROVAL CARD
# ---------------------------------------------------------------------------
# The graph pauses with interrupt() before a sensitive tool runs. The
# checkpoint keeps that pause, so this card is rebuilt from graph state on
# every rerun (and survives restarting Streamlit).

current_thread = st.session_state.get('current_thread')
pending_approval = get_pending_interrupt(current_thread) if current_thread else None

if pending_approval:
    gated_calls = [
        call for call in pending_approval.get('tool_calls', [])
        if call.get('requires_approval', True)
    ]

    with st.container(border=True):
        st.markdown('### 🛑 Approval required')
        st.caption(
            'The assistant wants to run a sensitive tool. '
            'Approve to let it continue, or reject to cancel the action.'
        )
        if pending_approval.get('question'):
            st.caption(f"Your request: “{pending_approval['question']}”")

        for call in gated_calls:
            st.markdown(f"**{tool_pill_label(call.get('name'))}**")
            st.code(
                json.dumps(call.get('args') or {}, indent=2, ensure_ascii=False),
                language='json',
            )

        reason = st.text_input(
            'Reason (optional)',
            key=f'approval_reason_{current_thread}',
        )
        approve_col, reject_col = st.columns(2)
        approve = approve_col.button(
            '✅ Approve',
            type='primary',
            use_container_width=True,
            key=f'approve_{current_thread}',
        )
        reject = reject_col.button(
            '❌ Reject',
            use_container_width=True,
            key=f'reject_{current_thread}',
        )

    if approve or reject:
        decision = (
            {'approved': True}
            if approve
            else {'approved': False, 'reason': reason.strip()}
        )
        config = thread_config(
            current_thread,
            st.session_state['thread_titles'].get(current_thread),
        )
        with st.spinner(
            'Running the approved action…' if approve else 'Cancelling the action…'
        ):
            ai_message = stream_assistant_reply(
                Command(resume=decision),
                config,
                preopened_calls=pending_approval.get('tool_calls', []),
            )

        if ai_message:
            st.session_state['message_history'].append(
                {'role': 'assistant', 'content': ai_message}
            )

        if (
            get_pending_interrupt(current_thread) is None
            and st.session_state['thread_titles'].get(current_thread) == 'New chat'
        ):
            with st.spinner('Naming this conversation...'):
                generate_title(current_thread)
        st.rerun()


# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------

# While an approval is pending the box is disabled so a second run cannot be
# started on top of the paused checkpoint.
chat_value = st.chat_input(
    'Message Agentic Chat...',
    accept_file='multiple',
    file_type=['pdf'],
    disabled=bool(pending_approval),
)

if chat_value:
    user_input = (chat_value.text or '').strip()
    uploaded_files = chat_value.files or []
    thread_id = st.session_state['current_thread']

    if uploaded_files:
        try:
            with st.spinner('Reading and indexing PDFs…'):
                paths = save_uploaded_pdfs(uploaded_files)
                stats = ingest_rag_document(paths)
        except Exception as exc:
            st.error(f'Could not index the PDF: {exc}')
            st.stop()

        st.session_state['kb_version'] = st.session_state.get('kb_version', 0) + 1
        st.session_state.pop('kb_signature', None)

        if not user_input:
            note = (
                f'📚 Indexed **{", ".join(file.name for file in uploaded_files)}** '
                f'({stats.get("chunks")} chunks). Ask me anything about it.'
            )
            st.session_state['message_history'].append(
                {'role': 'assistant', 'content': note}
            )
            with st.chat_message('assistant'):
                st.markdown(note)
            st.stop()

    if user_input:
        st.session_state['message_history'].append(
            {'role': 'user', 'content': user_input}
        )
        with st.chat_message('user'):
            st.text(user_input)
            if uploaded_files:
                st.caption('📎 ' + ', '.join(file.name for file in uploaded_files))

    if uploaded_files and user_input:
        st.caption(
            f'📚 Indexed {", ".join(file.name for file in uploaded_files)} '
            f'· {stats.get("chunks")} chunks'
        )

    if not user_input:
        st.stop()

    CONFIG = thread_config(thread_id, st.session_state['thread_titles'][thread_id])

    ai_message = stream_assistant_reply(
        {'messages': [HumanMessage(content=user_input)]},
        CONFIG,
    )

    if ai_message:
        st.session_state['message_history'].append(
            {'role': 'assistant', 'content': ai_message}
        )

    # The graph paused before a sensitive tool -> rerun so the approval card
    # above the chat box is rendered from the checkpoint.
    if get_pending_interrupt(thread_id) is not None:
        st.rerun()

    # After the first full exchange, name the conversation from its topic.
    if st.session_state['thread_titles'][thread_id] == 'New chat':
        with st.spinner('Naming this conversation...'):
            generate_title(thread_id)
        st.rerun()

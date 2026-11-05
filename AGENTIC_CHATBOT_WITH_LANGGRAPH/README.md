# Agentic Chatbot using LangGraph

A complete Streamlit chatbot backed by a LangGraph agent. The agent can search
the web, answer questions from uploaded PDFs, run calculations, look up stock
prices, and place **mock** stock purchase orders.

Sensitive tool calls use **human-in-the-loop (HITL)** approval: the graph pauses
with `interrupt()` and waits for a human to approve or reject the action before
the tool runs.

## Project structure

```
AGENTIC-CHATBOT-USING-LANGGRAPH/
├── .env
├── .gitignore
├── LICENSE
├── README.md
├── app.py          # Streamlit frontend
├── backend.py      # LangGraph agent, tools, RAG, SQLite persistence, HITL
└── requirements.txt
```

## Features

- **Persistent conversations** — checkpoints and thread titles are stored in
  `chatbot_conversations.db`.
- **ChatGPT-style sidebar** — create, reopen, rename, and delete conversations.
- **Streaming answers** — assistant tokens stream into the chat with collapsible
  tool activity pills.
- **PDF knowledge base (RAG)** — upload PDFs from the sidebar or the chat box;
  the FAISS index lives in `faiss_db/`.
- **Tools**
  - `rag_tool` — search uploaded PDFs
  - `search_tool` — Tavily web search
  - `calculator` — safe math evaluation
  - `get_stock_price` — Alpha Vantage quote lookup
  - `purchase_stock` — **mock** stock purchase (no real brokerage API)
- **Human-in-the-loop approval** — approve or reject gated tool calls from the
  Streamlit UI, or from the backend CLI.
- **Restart-safe HITL** — if the app stops while an approval is pending, the
  pending action is restored from the checkpoint.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure `.env`:

   ```dotenv
   GROQ_API_KEY=...
   TAVILY_API_KEY=...
   OPENAI_API_KEY=...            # optional, only for OpenAI models
   LANGSMITH_TRACING=false
   LANGSMITH_API_KEY=...         # optional
   HITL_TOOLS=purchase_stock     # default; see below
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. (Optional) Run the backend from the terminal:

   ```bash
   python backend.py
   ```

## Human-in-the-loop configuration

`HITL_TOOLS` controls which tools need approval before they execute:

| Value | Behaviour |
| --- | --- |
| `purchase_stock` | Only the mock stock purchase is gated (default) |
| `purchase_stock,get_stock_price` | Both tools are gated |
| `all` or `*` | Every tool call is gated |

When a gated tool is requested, the graph pauses and the frontend shows an
approval card with the tool name and arguments:

- **Approve** — resumes the graph with `Command(resume={"approved": True})` and
  runs the tool.
- **Reject** — resumes with `{"approved": False, "reason": "..."}`; the tool is
  not executed and the rejection is written to the conversation as a tool
  result so the message history stays valid.

The backend CLI prompts for the same decision:

```
⚠️  Human approval required
   • purchase_stock({"symbol": "AAPL", "quantity": 3})
Approve? [y/N]
```

## Test the HITL flow

In the Streamlit app, send:

```
Buy 3 shares of AAPL now.
```

The approval card appears before the mock order runs. Approve it to see the
confirmation payload, or reject it to cancel the order.

## Notes

- `purchase_stock` is a mock tool. It does **not** call a brokerage API and does
  not place a real trade.
- The Alpha Vantage demo key in `backend.py` is for demonstration only; use your
  own key for production use.
- `faiss_db/`, `documents/`, and the SQLite database are gitignored because they
  are generated at runtime.

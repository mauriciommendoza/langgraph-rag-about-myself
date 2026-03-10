# LangGraph RAG Agent

An intelligent question-answering agent powered by **LangGraph**, **LangChain**, and **Groq**. It uses Retrieval-Augmented Generation (RAG) to answer questions based on a curated knowledge base, with automatic fallback to web search when needed. Built with a modular, production-ready architecture.

---

## What Does This Project Do?

Imagine you have a collection of articles about AI topics (like prompt engineering, LLM agents, or adversarial attacks). This project builds an **autonomous agent** that:

1. **Reads your question** and decides the best way to answer it.
2. **Searches your knowledge base** (stored in Pinecone) for relevant information.
3. **Evaluates the quality** of what it found — are the documents actually useful?
4. **Falls back to the internet** (via Tavily) if the local documents aren't enough.
5. **Generates a concise answer** using Groq's LLM.
6. **Self-checks for hallucinations** — if the answer isn't grounded in facts, it retries automatically.

All of this happens in a **graph-based workflow** where each step is a node, and the agent dynamically decides which path to take based on the quality of intermediate results.

---

## Architecture Overview

```
                    ┌─────────────────┐
                    │   User Question │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Router      │  Decides: Vectorstore or Web?
                    └──┬──────────┬───┘
                       │          │
              ┌────────▼──┐  ┌───▼──────────┐
              │ Retrieve  │  │  Web Search  │
              │ (Pinecone)│  │  (Tavily)    │
              └────────┬──┘  └───┬──────────┘
                       │         │
              ┌────────▼──┐      │
              │  Grade    │      │
              │ Documents │      │
              └──┬─────┬──┘      │
                 │     │         │
          Relevant  Not Relevant │
                 │     └─────────┘
                 │
              ┌──▼──────────┐
              │   Generate  │  LLM creates the answer
              └──────┬──────┘
                     │
              ┌──────▼──────────────┐
              │ Hallucination Check │  Is it grounded in facts?
              └──┬────────┬────┬───┘
                 │        │    │
              Useful   Not   Not
                 │    Useful  Supported
                 │        │    │
              ┌──▼──┐    │    │
              │ END │    └────┘──► Retry
              └─────┘
```

---

## Tech Stack

| Component         | Technology                                    | Purpose                                      |
|--------------------|-----------------------------------------------|----------------------------------------------|
| **LLM**           | Groq (`llama-3.3-70b-versatile`)              | Language model for reasoning and generation   |
| **Orchestration** | LangGraph + LangChain                         | Graph-based agent workflow                    |
| **Vector Store**  | Pinecone (Serverless, cloud)                  | Persistent storage for document embeddings    |
| **Embeddings**    | HuggingFace (`all-MiniLM-L6-v2`, 384 dims)   | Converts text to numerical vectors            |
| **Web Search**    | Tavily Search API                             | Real-time internet search fallback            |
| **Package Manager**| uv                                           | Fast Python dependency management             |

---

## Project Structure

```
Langgraph-rag-psyco-oncology-chat/
│
├── main.py                          # Entry point — run this to use the agent
├── .env                             # API keys (not committed to git)
├── pyproject.toml                   # Project dependencies
│
└── src/
    ├── config/                      # Configuration and settings
    │   ├── settings.py              # URLs, chunk sizes, embedding model name
    │   └── llm.py                   # Groq LLM initialization
    │
    ├── retrieval/                   # Document indexing pipeline
    │   └── document_indexer.py      # Downloads, splits, and stores docs in Pinecone
    │
    ├── chains/                      # LangChain prompt chains
    │   ├── router.py                # Routes questions to vectorstore or web
    │   ├── generator.py             # RAG prompt that generates the final answer
    │   └── graders.py               # Three graders: relevance, hallucination, usefulness
    │
    └── graph/                       # LangGraph workflow
        ├── state.py                 # Defines the data structure passed between nodes
        ├── nodes.py                 # Node functions (retrieve, generate, grade, search)
        ├── edges.py                 # Conditional routing logic between nodes
        └── builder.py               # Assembles and compiles the full graph
```

---

## Detailed Module Breakdown

### `src/config/` — Configuration

- **`settings.py`** — Central place for all tunable parameters:
  - `EMBEDDING_MODEL`: Which HuggingFace model converts text to vectors (currently `all-MiniLM-L6-v2`).
  - `CHUNK_SIZE` / `CHUNK_OVERLAP`: How documents are split into smaller pieces for indexing.
  - `URLS_TO_LOAD`: The list of source articles that form the agent's knowledge base.

- **`llm.py`** — Initializes the Groq language model (`llama-3.3-70b-versatile`) using the API key from `.env`. All chains and graders in the project share this single LLM instance.

---

### `src/retrieval/` — Document Indexing

- **`document_indexer.py`** — The `DocumentIndexer` class handles the full indexing pipeline:
  1. **Downloads** articles from the configured URLs using `WebBaseLoader`.
  2. **Splits** them into small chunks (250 characters each) with `RecursiveCharacterTextSplitter`.
  3. **Embeds** each chunk into a 384-dimensional vector using HuggingFace.
  4. **Stores** all vectors in a **Pinecone** index in the cloud.

  **Smart indexing:** On every run, the indexer checks if the Pinecone index already has data. If it does, it skips the download and upload entirely — making subsequent runs much faster and avoiding duplicates.

---

### `src/chains/` — Prompt Chains

These are the "brains" of the agent — each one is a small LLM-powered pipeline that takes an input and returns a structured decision.

- **`router.py`** — The **Question Router**. Given a user question, it decides whether the answer is likely in the local vectorstore or if a web search is needed. Returns `{"datasource": "vectorstore"}` or `{"datasource": "web_search"}`.

- **`generator.py`** — The **RAG Generator**. Takes the retrieved context documents and the user's question, and generates a concise 3-sentence answer.

- **`graders.py`** — Contains **three separate graders**, each returning `{"score": "yes"}` or `{"score": "no"}`:
  - **Retrieval Grader**: Is this document relevant to the question?
  - **Hallucination Grader**: Is the generated answer grounded in the provided facts?
  - **Answer Grader**: Does the answer actually resolve the user's question?

---

### `src/graph/` — LangGraph Workflow

This is where all pieces come together into an autonomous decision-making graph.

- **`state.py`** — Defines `GraphState`, a typed dictionary that flows between every node. It carries the `question`, `documents`, `generation`, and a `web_search` flag.

- **`nodes.py`** — Contains four executable node functions:
  - `retrieve`: Queries Pinecone for relevant documents.
  - `grade_documents`: Runs each document through the Retrieval Grader and filters out irrelevant ones.
  - `web_search`: Calls the Tavily API for supplementary internet results.
  - `generate`: Invokes the RAG chain to produce the final answer.

- **`edges.py`** — Contains three conditional routing functions:
  - `route_question`: Entry point — vectorstore or web search?
  - `decide_to_generate`: After grading, are documents good enough or do we need web search?
  - `grade_generation_v_documents_and_question`: After generation, is the answer grounded and useful?

- **`builder.py`** — Wires all nodes and edges together into a `StateGraph`, compiles it, and exports the final `app` object.

---

### `main.py` — Entry Point

A lightweight CLI wrapper. It accepts a question as a command-line argument, streams the graph execution step-by-step, and prints the final answer in a clean format.

---

## Setup & Installation

### Prerequisites

- **Python 3.13+**
- **uv** (Python package manager) — [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- A **Groq** account → [console.groq.com](https://console.groq.com)
- A **Pinecone** account → [app.pinecone.io](https://app.pinecone.io)
- A **Tavily** account → [app.tavily.com](https://app.tavily.com)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Langgraph-rag-psyco-oncology-chat.git
cd Langgraph-rag-psyco-oncology-chat
```

### Step 2: Install Dependencies

```bash
uv sync
```

This reads `pyproject.toml` and installs everything into an isolated `.venv`.

### Step 3: Configure Environment Variables

Create a `.env` file in the project root (or edit the existing one) with your API keys:

```env
GROQ_API_KEY="your_groq_api_key_here"
HF_TOKEN="your_huggingface_token_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="psyco-oncology-rag"
TAVILY_API_KEY="your_tavily_api_key_here"
USER_AGENT="LangGraph-Agent"
```

| Variable             | Where to Get It                                      |
|----------------------|------------------------------------------------------|
| `GROQ_API_KEY`       | [Groq Console](https://console.groq.com/keys)       |
| `HF_TOKEN`           | [HuggingFace Settings](https://huggingface.co/settings/tokens) |
| `PINECONE_API_KEY`   | [Pinecone Console](https://app.pinecone.io)          |
| `PINECONE_INDEX_NAME`| Name you choose for your index (e.g. `psyco-oncology-rag`) |
| `TAVILY_API_KEY`     | [Tavily Dashboard](https://app.tavily.com)           |

> **Note:** The Pinecone index is created automatically on the first run if it doesn't exist. You don't need to create it manually.

### Step 4: Run the Agent

```bash
uv run main.py "Your question here"
```

**Examples:**

```bash
uv run main.py "What is agent memory?"
uv run main.py "Explain prompt engineering techniques"
uv run main.py "How to perform an adversarial attack on LLM?"
```

If no question is provided, it defaults to: *"What is agent memory?"*

---

## Example Output

```
============================================================
  LANGGRAPH RAG AGENT
  Question: What is agent memory?
============================================================
  [Pinecone] Index 'psyco-oncology-rag' has 187 vectors. Skipping upload.
  [Router] Deciding data source...
  [Router] -> Vectorstore (RAG)
  [Retrieve] Searching vectorstore for relevant documents...
  [Retrieve] Found 4 document(s).
  [+] Node completed: retrieve
  [Grader] Checking document relevance...
    Doc 1: Relevant
    Doc 2: Relevant
    Doc 3: Relevant
    Doc 4: Relevant
  [Grader] 4/4 documents passed.
  [Decision] Documents are sufficient -> Generate
  [+] Node completed: grade_documents
  [Generate] Generating answer from context...
  [Hallucination Check] Verifying answer is grounded in facts...
  [Hallucination Check] Answer is grounded. Checking usefulness...
  [Answer Check] Answer is useful!
  [+] Node completed: generate

------------------------------------------------------------
  FINAL ANSWER
------------------------------------------------------------

  Agent memory refers to the ability of an agent to retain and
  recall information over time...

============================================================
```

---

## Knowledge Base

The agent's knowledge base is built from these source articles (configurable in `src/config/settings.py`):

| Article                              | Topic                        |
|--------------------------------------|------------------------------|
| [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | Agent architectures & memory |
| [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) | Prompt techniques            |
| [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) | LLM security                 |

To add new sources, simply append URLs to the `URLS_TO_LOAD` list in `settings.py` and delete your Pinecone index (it will be recreated automatically on the next run).

---

## License

This project is for educational and research purposes.
# DriveThruData — Multi-RAG Orchestration System

A modular **Retrieval-Augmented Generation** framework that implements six different RAG strategies and intelligently routes queries to the optimal one using an LLM-based intent classifier.

---

## Strategies

| # | Strategy | File | Mechanism |
|---|----------|------|-----------|
| 1 | **Normal RAG** | `strategies/normal_rag.py` | Dense vector search on Pinecone (`top_k=5`), `nomic-embed-text` embeddings |
| 2 | **Hybrid RAG** | `strategies/hybrid_rag.py` | Dense + sparse search (each `top_k=20`), deduplication, Pinecone `bge-reranker-v2-m3` reranker (`top_n=10`) |
| 3 | **Corrective RAG** | `strategies/corrective_rag.py` | Wraps HybridRAG + LLM document grading → falls back to wider dense search if no relevant docs |
| 4 | **Graph RAG** | `strategies/graph_rag.py` | Neo4j knowledge graph — Cypher `CONTAINS` match on node names, returns `source-relationship->target` triples |
| 5 | **Cache RAG** | `strategies/cache_rag.py` | Semantic cache — in-memory cosine similarity against past queries (threshold 0.9); falls through to NormalRAG on miss |
| 6 | **Agentic RAG** | `strategies/agentic_rag.py` | Multi-step reasoning loop (up to 3 iterations): LLM thinks → sub-query → HybridRAG retrieve → synthesize |

---

## Architecture

```
User Input (CLI / Web UI)
        │
        ▼
  RAGOrchestrator.route_query()
        │
        ├─ 1. CacheRAG check (semantic hit?)
        │      └─ miss → continue
        │
        ├─ 2. LLM Intent Classifier
        │      └─ routes to: normal | hybrid | corrective | graph | agentic
        │
        ▼
  Strategy.retrieve(query) ──→ Strategy.generate(query, context) ──→ Response
```

---

## Directory Layout

```
DriveThruData/
├── main.py                    # CLI entry point
├── app.py                     # Flask web app
│
├── core/
│   ├── base_rag.py            # Abstract BaseRAG class
│   ├── config.py              # Config (Pinecone, Neo4j, Ollama)
│   └── orchestrator.py        # RAGOrchestrator — query routing
│
├── strategies/
│   ├── normal_rag.py
│   ├── hybrid_rag.py
│   ├── corrective_rag.py
│   ├── graph_rag.py
│   ├── cache_rag.py
│   └── agentic_rag.py
│
├── static/
│   ├── style.css              # Multi-RAG Explorer UI styles
│   └── script.js              # Client-side logic (strategy switching, chat)
│
├── templates/
│   └── index.html             # Flask Jinja2 template
│
├── Corrective_RAG/            # Standalone CRAG (Flask UI)
│   ├── main.py / app.py
│   ├── sample.txt
│   └── static/ + templates/
│
├── CRAG/                      # Duplicate of Corrective_RAG/
│
├── RAG_PIPELINE_FINAL/        # Production-oriented variant (Redis cache, cloud Ollama)
│   ├── main.py
│   ├── test.py
│   └── sample.txt
│
├── rag_pipeline_zero_hallucination.ipynb  # 10-block zero-hallucination RAG notebook
│
├── AGENTIC RAG.json           # n8n workflow — Agentic RAG
├── CACHE RAG.json             # n8n workflow — Cache RAG
├── Vectorless RAG.json        # n8n workflow — SQL/MongoDB/Redis routing
│
├── .env                       # Credentials (Pinecone, Neo4j, Ollama)
├── .gitignore
└── .gitmodules
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with models: `llama3`, `nomic-embed-text`, `gemma4:31b`, `ministral-3:14b`
- Pinecone account (API key + index host)
- Neo4j instance (URI, user, password)
- Redis server (for RAG_PIPELINE_FINAL)

### Setup

```bash
# Clone with submodules
git clone --recursive <repo-url> DriveThruData
cd DriveThruData

# Python environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install flask pinecone-client neo4j redis ollama chromadb

# Configure environment
# Edit .env with your keys
```

### Run

```bash
# CLI mode
python main.py

# Web UI mode
python app.py
# → http://localhost:5000
```

---

## Standalone Variants

### `Corrective_RAG/` & `CRAG/`
Self-contained CRAG pipeline with its own Flask chat UI.
- `main.py` — CLI: reads `sample.txt`, creates Pinecone indices, interactive Q&A with grading & fallback
- `app.py` — Web UI at `/chat`

### `RAG_PIPELINE_FINAL/`
Production-oriented variant with:
- Remote Ollama (cloud API) instead of local
- Redis response + chunk caching (24-hour TTL)
- Separate Pinecone indices (`dense-vectors-final`, `sparse-vectors-final`)
- Model: `ministral-3:14b`

---

## n8n Workflows

The `.json` files are importable into [n8n](https://n8n.io/) for visual pipeline editing:

| File | Routes | Vector Store | LLM |
|------|--------|--------------|-----|
| `AGENTIC RAG.json` | SQL / MongoDB / Redis / Vector | ChromaDB (Gemini embeddings) | gemma4:31b |
| `CACHE RAG.json` | ChromaDB + Redis memory | ChromaDB | gemma4:31b |
| `Vectorless RAG.json` | SQL (Supabase) / MongoDB / Redis | None | gemma4:31b |

---

## Zero-Hallucination Notebook

`rag_pipeline_zero_hallucination.ipynb` — 1376-line Jupyter notebook covering:

1. Ingestion & chunking
2. Hybrid retrieval (BM25 + embeddings)
3. ANN search + reranking
4. Source confidence scoring
5. Constrained generation
6. Citation-backed responses
7. Confidence threshold gating
8. Continuous evaluation
9. Caching & memory
10. Observability

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Web Framework | Flask |
| Vector DB | Pinecone (dense + sparse) |
| Graph DB | Neo4j |
| Cache | Redis / In-memory dict |
| LLMs | Ollama (local) / Ollama Cloud |
| Models | llama3, gemma4:31b, ministral-3:14b, nomic-embed-text |
| Orchestration | n8n (JSON workflows) |
| Frontend | HTML, CSS, JavaScript |

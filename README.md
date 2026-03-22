# Agentic Bot Builder

A Python project that takes any knowledge source — a website, a PDF, or plain text — and turns it into a fully functional AI-powered chatbot that can answer questions, cite its sources, and handle gaps in its knowledge gracefully. Built to explore how RAG pipelines, vector databases, and multi-agent orchestration work together in a real application.

## What This Project Does

The idea is straightforward. You create a bot, feed it knowledge, and it answers questions. But under the hood, quite a lot is happening.

1. You define a bot with a name, domain, and persona through the API
2. You ingest knowledge into it — scrape a URL, upload a PDF, or paste raw text
3. The ingestion pipeline chunks the text into overlapping segments and embeds them using a local sentence transformer model
4. Those embeddings are stored in a ChromaDB vector collection that is isolated per bot
5. When a user asks a question, a LangGraph agent pipeline takes over
6. The query is first analyzed and rewritten for better retrieval by a query router agent
7. A RAG retrieval agent searches the vector store using cosine similarity and scores the results
8. If confidence is high enough, an answer generator agent sends the retrieved context to Claude and gets a grounded answer back
9. If confidence is too low, a fallback agent does a broader search and generates a helpful response pointing to related topics
10. A citation agent then structures all source references and attaches them to the response
11. The final response includes the answer, confidence score, source links, related topics, and a full agent trace showing which agents ran

## Tools and Libraries Used

**FastAPI** — the web framework powering the REST API. Chosen for its async support, automatic Swagger documentation, and Pydantic integration. Every endpoint is fully documented and testable through the browser at `/docs` without writing any additional code.

**LangGraph** — used to build the multi-agent pipeline as a state machine. Each agent is a node in a directed graph, and routing between nodes is conditional based on retrieval confidence. This makes the pipeline easy to extend — adding a new agent is just adding a new node and connecting it with edges.

**LangChain Anthropic** — the integration layer between LangGraph and Claude. The `ChatAnthropic` class handles prompt formatting, API calls, and response parsing. Temperature is set to 0.3 to keep answers factual and consistent rather than creative.

**Anthropic Claude (claude-3-5-sonnet-20241022)** — the language model used for query rewriting, answer generation, and fallback responses. Claude was chosen for its strong instruction-following and its ability to stay grounded in provided context without hallucinating when told not to.

**ChromaDB** — the vector database that stores all embeddings. Each bot gets its own isolated collection named `bot_{id}`, so bots never share or contaminate each other's knowledge. ChromaDB runs locally and persists to disk, so embeddings survive server restarts without needing to re-ingest.

**Sentence Transformers (all-MiniLM-L6-v2)** — used to generate embeddings locally without any API cost. This 384-dimension model is fast, lightweight, and produces good semantic similarity results for retrieval tasks. The model downloads automatically on first use and is cached locally afterward.

**Why cosine similarity specifically?** — cosine similarity measures the angle between two vectors rather than their magnitude, which makes it robust to differences in text length. A short question and a long paragraph can still match well if they are about the same topic. ChromaDB's HNSW index uses cosine distance by default when the collection is created with `hnsw:space: cosine`.

**SQLAlchemy with aiosqlite** — used for storing bot configurations, knowledge source metadata, and chat session history. SQLite was chosen over a heavier database like PostgreSQL because it requires zero setup, runs entirely as a local file, and is more than sufficient for the scale this project targets. The async driver means database operations never block the API.

**PyPDF2** — used to extract text from uploaded PDF files page by page. It handles most standard PDFs well. For PDFs with complex layouts, tables, or scanned images, PyMuPDF would be a better alternative as it has stronger layout detection.

**BeautifulSoup4 with httpx** — used for web scraping. httpx is used instead of requests because it supports async HTTP calls natively, which means multiple URLs can be scraped without blocking. BeautifulSoup strips out navigation bars, footers, scripts, and other non-content elements to extract clean readable text.

**Pydantic Settings** — all configuration is typed and validated through a `Settings` class that reads from the `.env` file. This means the app fails immediately on startup with a clear error if a required variable like `ANTHROPIC_API_KEY` is missing, rather than failing silently later during a request.

**Why `@lru_cache` on `get_settings()`?** — the `.env` file is read once at startup and cached. Every subsequent call to `get_settings()` returns the same object from memory rather than re-reading the file. This is a small but important optimization since settings are accessed on every request.

## Project Structure
```
Bot-Builder/
├── app/
│   ├── agents/
│   │   └── graph.py          ← LangGraph multi-agent pipeline
│   ├── api/
│   │   └── routes/
│   │       ├── bots.py       ← Bot CRUD endpoints
│   │       ├── ingestion.py  ← URL / PDF / text ingestion endpoints
│   │       └── chat.py       ← Chat endpoint
│   ├── core/
│   │   ├── config.py         ← Settings and environment config
│   │   ├── database.py       ← SQLAlchemy async database setup
│   │   └── vector_store.py   ← ChromaDB manager and embeddings
│   ├── ingestion/
│   │   └── pipeline.py       ← Web scraper, PDF parser, text chunker
│   ├── models/
│   │   └── schemas.py        ← All Pydantic request and response models
│   └── main.py               ← FastAPI app entry point and lifespan
├── data/
│   ├── chroma_db/            ← Vector embeddings (auto-created)
│   └── uploads/              ← Uploaded files (auto-created)
├── tests/
│   └── __init__.py
├── .env.example              ← Environment variable template
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to Run It

**Step 1 — Clone the repository**
```bash
git clone https://github.com/SarveshMohan89/Agentic-Bot-Builder.git
cd Agentic-Bot-Builder
```

**Step 2 — Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```
This will take 3-5 minutes. sentence-transformers and chromadb are large packages.

**Step 4 — Set up environment variables**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-3-5-sonnet-20241022
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.35
DATABASE_URL=sqlite+aiosqlite:///./data/agentic_bots.db
MAX_UPLOAD_SIZE_MB=50
UPLOAD_DIR=./data/uploads
```

Get your Anthropic API key from https://console.anthropic.com

**Step 5 — Start the server**
```bash
python -m uvicorn app.main:app --reload
```

**Step 6 — Open the interactive docs**

Visit http://localhost:8000/docs in your browser. Every endpoint is listed and testable directly from the browser.

## API Overview
```
POST   /api/v1/bots                              Create a new bot
GET    /api/v1/bots                              List all bots
GET    /api/v1/bots/{bot_id}                     Get a bot by ID
PATCH  /api/v1/bots/{bot_id}                     Update bot config
DELETE /api/v1/bots/{bot_id}                     Delete bot and its data
GET    /api/v1/bots/{bot_id}/stats               Vector store stats

POST   /api/v1/bots/{bot_id}/ingest/urls         Scrape and ingest URLs
POST   /api/v1/bots/{bot_id}/ingest/pdf          Upload and ingest a PDF
POST   /api/v1/bots/{bot_id}/ingest/text         Ingest raw text
GET    /api/v1/bots/{bot_id}/sources             List knowledge sources
DELETE /api/v1/bots/{bot_id}/sources/{source_id} Remove a source

POST   /api/v1/bots/{bot_id}/chat                Ask the bot a question
GET    /api/v1/bots/{bot_id}/sessions/{id}/history  Get chat history
DELETE /api/v1/bots/{bot_id}/sessions/{id}       Clear a session
```

## How the Agent Pipeline Works
```
User Question
     │
     ▼
QueryRouterAgent      Rewrites the query for better retrieval
     │                and classifies intent
     ▼
RAGRetrievalAgent     Searches ChromaDB using cosine similarity
     │                and scores the top results
     │
     ├── confidence HIGH ──► AnswerGeneratorAgent
     │                       Sends context to Claude
     │                       Gets a grounded answer back
     │
     └── confidence LOW  ──► FallbackAgent
                             Broader search, finds related topics
                             Generates a helpful redirect response
                             │
                             ▼
                       CitationAgent
                       Deduplicates sources
                       Structures reference links
                       Attaches related topics
                             │
                             ▼
                        Final Response
```

## Example Response
```json
{
  "session_id": "test-session-001",
  "question": "What is machine learning?",
  "answer": "Machine learning is a subset of AI that enables systems
             to learn from data without being explicitly programmed.",
  "confidence": 0.85,
  "answer_type": "direct",
  "sources": [
    {
      "title": "What is Artificial Intelligence",
      "url": "https://example.com/what-is-ai",
      "snippet": "Machine learning is a subset of AI...",
      "relevance_score": 0.85,
      "source_type": "text"
    }
  ],
  "related_topics": [],
  "agent_trace": [
    "QueryRouterAgent",
    "RAGRetrievalAgent",
    "AnswerGeneratorAgent",
    "CitationAgent"
  ],
  "processing_time_ms": 1842
}
```

## A Note on Answer Quality

The answers this project generates are only as good as the knowledge that has been ingested. If the ingested content does not contain information relevant to the question, the bot will route to the fallback agent and return related topics instead of a direct answer. This is by design — the bot is strictly grounded in its knowledge base and will not make things up.

Web scraping quality varies by site. Some sites block scrapers or use JavaScript rendering that httpx cannot handle. For those cases, copying the text and using the `/ingest/text` endpoint directly will always work.

The similarity threshold of 0.35 is a reasonable default but may need tuning depending on your content. Lowering it makes the bot more likely to attempt an answer but risks less relevant context. Raising it makes answers more precise but increases fallback responses.

## What Could Be Better

**Streaming responses** — currently the entire answer is generated before being sent back. Adding server-sent events would allow the answer to stream token by token, making the experience feel much faster especially for longer answers.

**A frontend UI** — right now everything is accessed through the Swagger docs. A React dashboard with a bot creation wizard, knowledge base manager, and chat interface would make this usable for non-technical users.

**Re-ranking** — after the initial vector search, a cross-encoder re-ranker could score the retrieved chunks more accurately before passing them to Claude. This would improve answer quality especially when the top retrieved chunks are only loosely related to the question.

**Smarter chunking** — the current approach splits by sentence count with overlap. Splitting by semantic similarity or by document headings would produce more coherent chunks and therefore better retrieval results.

**Analytics** — tracking query confidence scores, fallback rates, and response times over time would help identify gaps in the knowledge base and measure how well the bot is performing.

**Authentication** — there is currently no API key or user authentication. Adding per-bot API keys would make this ready for multi-client deployment where different teams manage their own bots independently.
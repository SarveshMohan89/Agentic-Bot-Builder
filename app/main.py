import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import init_db
from app.api.routes import bots, ingestion, chat

settings = get_settings()

@asynccontextmanager
async def lifespan(app : FastAPI):
    print("\n" + "=" *60)
    print(" Agentic Bot - Starting Up")
    print("=" *60)

    os.makedirs("./data/chroma_db", exist_ok=True)
    os.makedirs("./data/uploads", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    await init_db()

    from app.core.vector_store import get_vector_store
    get_vector_store()
    print(" LangGraph agent pipeline compiled! ")

    print("=" *60)
    print(f" API running at http://{settings.app_host}:{settings.app_port}")
    print(f" Docs at http://localhost:{settings.app_port}/docs")
    print("=" *60 + "\n")

    yield

    print("\n Agentic Bot shutting down... ")

app = FastAPI(
    title= "Agentic Bot API",
    description= (
        "A multi-tenant bot-building platform powered by "
        "LangGraph and Claude. Create intelligent virtual "
        "assistants with RAG, citations, and fallback handling."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

API_PREFIX = "/api/v1"

app.include_router(bots.router, prefix=API_PREFIX)
app.include_router(ingestion.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)

@app.get("/", tags = ["Health"])
async def root():
    return {
        "name" : settings.app_name,
        "version" : "1.0.0",
        "status" : "running",
        "docs" : "/docs"
    }

@app.get("/health", tags = ["Health"])
async def health():
    return {
        "status" : "healthy",
        "environment" : settings.app_env,
    }
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid

class BotCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    domain: str = Field(..., description= "Topic the bot covers")
    welcome_message: Optional[str] = Field(default="Hey There! I am Finzo, your smart assistant. How can I help you today?")
    system_prompt: Optional[str] = None
    fallback_message: Optional[str] = Field(default="Sorry, I could not find the exact answer to your question, please check some related topics which might help you.")
    config_meta: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BotUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    welcome_message: Optional[str] = None
    system_prompt: Optional[str] = None
    fallback_message: Optional[str] = None
    is_active: Optional[bool] = None
    config_meta: Optional[Dict[str, Any]] = None

class BotResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    domain: str
    welcome_message: Optional[str]
    system_prompt: Optional[str]
    fallback_message: Optional[str]
    is_active: bool
    config_meta: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class BotListResponse(BaseModel):
    total : int
    bots : List[BotResponse]

class URLIngestRequest(BaseModel):
    urls : List[str] = Field(..., min_length=1)
    crawl_depth : int = Field(default=1, ge=1, le=1)

class TextIngestRequest(BaseModel):
    title : str = Field(..., description="Title for this knowledge entry")
    content : str = Field(..., min_length=5)
    source_url : Optional[str] = None
    metadata : Optional[Dict[str, Any]] = Field(default_factory=dict)

class IngestJobResponse(BaseModel):
    job_id: str
    bot_id: str
    source_type: str
    source_name: str
    status: str
    chunk_count: int
    message: str
    created_at: datetime

    class Config:
        from_attributes = True

class KnowledgeSourceResponse(BaseModel):
    id: str
    bot_id: str
    source_type: str
    source_name: str
    source_url: Optional[str]
    chunk_count: int
    status: str
    error_message: Optional[str]
    source_meta: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    role : Literal["user", "assistant"] = "user"
    content : str

class ChatRequest(BaseModel):
    question : str = Field(..., min_length=1, max_length=2000)
    session_id : Optional[str] = Field(default_factory=lambda : str(uuid.uuid4()))
    history : Optional[List[ChatMessage]] = Field(default_factory=list)

class SourceReference(BaseModel):
    title : str
    url : Optional[str]
    snippet : str
    relevance_score : float
    source_type : str

class RelatedTopic(BaseModel):
    title : str
    url : Optional[str]
    snippet : str
    relevance_score : float

class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    confidence: float
    answer_type: Literal["direct", "partial", "fallback"]
    sources: List[SourceReference]
    related_topics: List[RelatedTopic]
    agent_trace: Optional[List[str]] = None
    processing_time_ms: int

class AgentState(BaseModel):
    bot_id: str
    question: str
    session_id: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    bot_config: Optional[Dict[str, Any]] = None

    # Document Retrival
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    related_docs: List[Dict[str, Any]] = Field(default_factory=list)

    # Classification
    query_intent: Optional[str] = None
    retrieval_confidence: float = 0.0

    # Generation
    answer: Optional[str] = None
    answer_type: str = "fallback"
    confidence: float = 0.0

    # Citations
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    related_topics: List[Dict[str, Any]] = Field(default_factory=list)

    # Trace
    agent_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = None

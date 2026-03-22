import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.database import get_db, BotConfig, ChatSession
from app.agents.graph import get_agent_graph
from app.models.schemas import (
AgentState, ChatRequest, ChatResponse,
SourceReference, RelatedTopic, ChatMessage
)
router = APIRouter(tags=["Chat"])

@router.post("/bots/{bot_id}/chat", response_model=ChatResponse)
async def chat(
        bot_id : str,
        payload : ChatRequest,
        db : AsyncSession = Depends(get_db),
):
    start_time = time.time()

    result = await db.execute(
        select(BotConfig).where(BotConfig.id == bot_id)
    )
    bot = result.scalar_one_or_none()
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot '{bot_id}' not found.",
        )

    if not bot.is_active:
        raise HTTPException(
            status_code= status.HTTP_403_FORBIDDEN,
            detail= "This bot is not active.",
        )

    session_id = payload.session_id or str(uuid.uuid4())

    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.bot_id == bot_id,
            ChatSession.session_id == session_id,
        )
    )
    session_record = session_result.scalar_one_or_none()

    history = []
    if session_record and session_record.messages:
        history = session_record.messages[-8:]

    elif payload.history:
        history = [m.model_dump() for m in payload.history]

    initial_state = AgentState(
        bot_id = bot_id,
        question = payload.question,
        session_id = session_id,
        chat_history = history,
        bot_config = {
            "domain" : bot.domain,
            "system_prompt" : bot.system_prompt,
            "fallback_message" : bot.fallback_message,
            "welcome_message" : bot.welcome_message,
        },
    )

    graph = get_agent_graph()
    try:
        final_state = await graph.ainvoke(initial_state)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline failed: {str(e)}",
        )

    new_messages = history + [
        {"role" : "user", "content" : payload.question},
        {"role" : "assistant", "content" : final_state.get("answer") or ""},
    ]

    if session_record:
        session_record.messages = new_messages

    else:
        new_session = ChatSession(
            bot_id = bot_id,
            session_id = session_id,
            messages = new_messages,
        )
        db.add(new_session)

    await db.flush()

    elapsed_ms = int((time.time() - start_time) *1000)

    sources = [
        SourceReference(
            title = s.get("title", "Source"),
            url = s.get("url"),
            snippet = s.get("snippet", ""),
            relevance_score = s.get("relevance_score", 0.0),
            source_type = s.get("source_type", "unknown"),
        )
        for s in (final_state.get("sources") or [])
    ]

    related_topics = [
        RelatedTopic(
            title = r.get("title", "Related"),
            url = r.get("url"),
            snippet = r.get("snippet", ""),
            relevance_score = r.get("relevance_score", 0.0),
        )
        for r in (final_state.get("related_topics") or [])
    ]

    return ChatResponse(
        session_id = session_id,
        question = payload.question,
        answer = final_state.get("answer") or bot.fallback_message,
        confidence = final_state.get("confidence") or 0.0,
        answer_type = final_state.get("answer_type") or "fallback",
        sources = sources,
        related_topics = related_topics,
        agent_trace = final_state.get("agent_trace"),
        processing_time_ms = elapsed_ms,
    )

@router.get(
    "/bots/{bot_id}/sessions/{session_id}/history",
    response_model=List[ChatMessage],

)
async def get_chat_history(
        bot_id : str,
        sesson_id : str,
        db : AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.bot_id == bot_id,
            ChatSession.session_id == session_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        return []
    return [ChatMessage(**m) for m in session.messages]

@router.delete(
    "/bots/{bot_id}/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_session(
        bot_id : str,
        session_id : str,
        db: AsyncSession = Depends(get_db),
):
    await db.execute(
        delete(ChatSession).where(
            ChatSession.bot_id == bot_id,
            ChatSession.session_id == session_id,
        )
    )
    return None
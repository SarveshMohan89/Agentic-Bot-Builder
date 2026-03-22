from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import List
import uuid

from app.core.database import get_db, BotConfig
from app.core.vector_store import get_vector_store
from app.models.schemas import (BotResponse, BotCreateRequest, BotUpdateRequest, BotListResponse,)

router = APIRouter(prefix="/bots", tags=["Bot Management"])

@router.post("", response_model=BotResponse, status_code=status.HTTP_201_CREATED)
async def create_bot(
        payload : BotCreateRequest,
        db : AsyncSession = Depends(get_db)
):
    bot = BotConfig(
        id = str(uuid.uuid4()),
        name = payload.name,
        description = payload.description,
        domain = payload.domain,
        welcome_message = payload.welcome_message,
        system_prompt = payload.system_prompt or _default_system_prompt(payload.name, payload.domain),
        fallback_message = payload.fallback_message,
        config_meta = payload.config_meta or {},
    )
    db.add(bot)
    await db.flush()
    await db.refresh(bot)
    return BotResponse.model_validate(bot)

@router.get("", response_model=BotListResponse)
async def list_bots(
        skip: int = 0,
        limit: int = 50,
        db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(BotConfig)
        .order_by(BotConfig.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    bots = result.scalars().all()

    count_result = await db.execute(select(BotConfig))
    total = len(count_result.scalars().all())

    return BotListResponse(
        total = total,
        bots = [BotResponse.model_validate(b) for b in bots],
    )

@router.get("/{bot_id}", response_model= BotResponse)
async def get_bot(
        bot_id : str,
        db : AsyncSession = Depends(get_db)
):
    bot = await _get_bot_or_404(bot_id, db)
    return BotResponse.model_validate(bot)

@router.patch("/{bot_id}", response_model=BotResponse)
async def update_bot(
        bot_id : str,
        payload : BotUpdateRequest,
        db : AsyncSession = Depends(get_db),
):
    bot = await _get_bot_or_404(bot_id, db)
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(bot, field, value)

    await db.flush()
    await db.refresh(bot)
    return BotResponse.model_validate(bot)

@router.delete("/{bot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bot(
        bot_id : str,
        db : AsyncSession = Depends(get_db)
):
    await _get_bot_or_404(bot_id, db)
    vector_store = get_vector_store()
    vector_store.delete_collection(bot_id)
    await db.execute(delete(BotConfig).where(BotConfig.id == bot_id))
    return None

@router.get("/{bot_id}/stats")
async def get_bot_stats(
        bot_id : str,
        db: AsyncSession = Depends(get_db)
):
    await _get_bot_or_404(bot_id, db)
    vector_store = get_vector_store()
    stats = vector_store.get_collection_stats(bot_id)
    return stats

async def _get_bot_or_404(bot_id : str, db : AsyncSession) -> BotConfig:
    result = await db.execute(
        select(BotConfig).where(BotConfig.id == bot_id)
    )
    bot = result.scalar_one_or_none()
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot with id '{bot_id}' not found."
        )
    return bot

def _default_system_prompt(bot_name : str, domain : str) -> str:
    return (
        f"You are {bot_name}, an intelligent assistant "
        f"specializing in {domain}. "
        f"You provide accurate, helpful and concise answers "
        f"based on your knowledge base. "
        f"Please always be friendly and professional."
    )
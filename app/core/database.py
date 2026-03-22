from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, DateTime, Text, Row, Boolean, JSON, Integer
from datetime import datetime, timezone
import uuid

from app.core.config import get_settings
settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo = settings.debug,
    connect_args = {"check_same_thread" : False},
)

AsyncSessionLocal = async_sessionmaker(
    bind = engine,
    class_ = AsyncSession,
    expire_on_commit = False,
)

class Base(DeclarativeBase):
    pass

class BotConfig(Base):
    __tablename__ = "bot_configs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    domain = Column(String(255), nullable=False)
    welcome_message = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=True)
    fallback_message = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    config_meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bot_id = Column(String, nullable=False, index=True)
    source_type = Column(String(50), nullable=False)
    source_name = Column(String(500), nullable=False)
    source_url = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    status = Column(String(50), default="pending")
    error_message = Column(Text, nullable=True)
    source_meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bot_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    messages = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()

        except Exception:
            await session.rollback()
            raise

        finally:
            await session.close()

async def init_db():
    import os
    os.makedirs("./data", exist_ok = True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("Tables initialized")
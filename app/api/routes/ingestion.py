import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, status, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.database import get_db, BotConfig, KnowledgeSource
from app.core.vector_store import get_vector_store
from app.core.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import (
URLIngestRequest,
TextIngestRequest,
IngestJobResponse,
KnowledgeSourceResponse,
)

router = APIRouter(tags = ["Knowledge Ingestion"])
settings = get_settings()
_pipeline = IngestionPipeline()

@router.post(
    "/bots/{bot_id}/ingest/urls",
    response_model= List[IngestJobResponse]
)

async def ingest_urls(
        bot_id : str,
        payload : URLIngestRequest,
        db : AsyncSession = Depends(get_db),
):
    await _assert_bot_exists(bot_id, id)

    results = []
    urls_to_process = list(payload.urls)

    if payload.crawl_depth > 1 and len(payload.urls) == 1:
        try:
            discovered = await _pipeline.scraper.discover_links(
                payload.urls[0], depth= payload.crawl_depth - 1
            )
            urls_to_process = list(set(urls_to_process + discovered))

        except Exception:
            pass

    for url in urls_to_process:
        source_id = str(uuid.uuid4())
        source_record = KnowledgeSource(
            id = source_id,
            bot_id = bot_id,
            source_type = "url",
            source_name = url,
            source_url = url,
            status = "processing",
        )
        db.add(source_record)
        await db.flush()

        try:
            result = await _pipeline.ingest_url(
                bot_id = bot_id,
                url = url,
                source_id= source_id,
            )
            source_record.status = "done"
            source_record.source_name = result["source_name"]
            source_record.chunk_count = result["chunk_count"]
            source_record.source_url = result["source_url"]

        except Exception as e:
            source_record.status = "Failed"
            source_record.error_message = str(e)

        await db.flush()
        await db.refresh(source_record)
        results.append(_to_ingest_response(source_record))

    return results

@router.post(
    "/bots/{bot_id}/ingest/pdf",
    response_model= IngestJobResponse
)
async def ingest_pdf(
        bot_id : str,
        file : UploadFile = File(...),
        source_url : str = Form(default=""),
        db : AsyncSession = Depends(get_db),
):
    await _assert_bot_exists(bot_id, db)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF Files are supported.",
        )
    file_bytes = await file.read()

    if len(file_bytes) > settings.max_uplaod_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb}MB,",
        )

    source_id = str(uuid.uuid4())
    source_record = KnowledgeSource(
        id = source_id,
        bot_id = bot_id,
        source_type = "pdf",
        source_name = file.filename,
        source_url = source_url or None,
        status = "processing",
    )
    db.add(source_record)
    await db.flush()

    try:
        result = await _pipeline.ingest_pdf(
            bot_id = bot_id,
            file_bytes= file_bytes,
            filename= file.filename,
            source_id= source_id,
            source_url= source_url or None,
        )
        source_record.status = "Done"
        source_record.source_name = result["source_name"]
        source_record.chunk_count = result["chunk_count"]

    except Exception as e:
        source_record.status = "Failed"
        source_record.error_message = str(e)
        raise HTTPException(
            status_code= status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process PDF: {e}",
        )
    finally:
        await db.flush()
        await db.refresh(source_record)

    return _to_ingest_response(source_record)

@router.post(
    "/bots/{bot_id}/ingest/text",
    response_model=IngestJobResponse
)
async def ingest_text(
    bot_id: str,
    payload: TextIngestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Ingest a block of plain text into the bot's knowledge base."""
    await _assert_bot_exists(bot_id, db)

    source_id = str(uuid.uuid4())
    source_record = KnowledgeSource(
        id=source_id,
        bot_id=bot_id,
        source_type="text",
        source_name=payload.title,
        source_url=payload.source_url,
        status="processing",
    )
    db.add(source_record)
    await db.flush()

    try:
        result = await _pipeline.ingest_text(
            bot_id=bot_id,
            title=payload.title,
            content=payload.content,
            source_id=source_id,
            source_url=payload.source_url,
            extra_metadata=payload.metadata,
        )
        source_record.status = "done"
        source_record.chunk_count = result["chunk_count"]
    except Exception as e:
        source_record.status = "failed"
        source_record.error_message = str(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to ingest text: {e}",
        )
    finally:
        await db.flush()
        await db.refresh(source_record)

    return _to_ingest_response(source_record)

@router.get(
    "/bots/{bot_id}/sources",
    response_model=List[KnowledgeSourceResponse]
)
async def list_sources(
        bot_id : str,
        db : AsyncSession = Depends(get_db)
):
    await _assert_bot_exists(bot_id, db)
    result = await db.execute(
        select(KnowledgeSource)
        .where(KnowledgeSource.bot_id == bot_id)
        .order_by(KnowledgeSource.created_at.desc())
    )
    sources = result.scalars().all()
    return [KnowledgeSourceResponse.model_validate(s) for s in sources]

@router.delete(
    "/bots/{bot_id}/sources/{source_id}",
    status_code= status.HTTP_204_NO_CONTENT,
)
async def delete_source(
        bot_id : str,
        source_id : str,
        db : AsyncSession = Depends(get_db),
):
    await _assert_bot_exists(bot_id, db)
    vector_store = get_vector_store()
    vector_store.delete_source(bot_id, source_id)
    await db.execute(
        delete((KnowledgeSource).where(
            KnowledgeSource.id == source_id,
            KnowledgeSource.bot_id == bot_id,
        ))
    )
    return None

async def _assert_bot_exists(bot_id : str, db : AsyncSession):
    result = await db.execute(
        select(BotConfig).where(BotConfig.id == bot_id)
    )
    bot = result.scalar_one_or_none()
    if not bot:
        raise HTTPException(
            status_code= status.HTTP_404_NOT_FOUND,
            detail=f"Bot '{bot_id}' not found",
        )
    return bot

def _to_ingest_response(source: KnowledgeSource) -> IngestJobResponse:
    return IngestJobResponse(
        job_id=source.id,
        bot_id=source.bot_id,
        source_type=source.source_type,
        source_name=source.source_name,
        status=source.status,
        chunk_count=source.chunk_count,
        message=(
            f"Successfully ingested {source.chunk_count} chunks"
            if source.status == "done"
            else source.error_message or "Processing failed"
        ),
        created_at=source.created_at,
    )

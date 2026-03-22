import re
import io
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.core.vector_store import get_vector_store

settings = get_settings()


class SemanticChunker:

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def split(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_len = overlap_len
            current_chunk.append(sentence)
            current_len += sentence_len
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return [c for c in chunks if len(c.strip()) > 20]


class WebScraper:

    SKIP_TAGS = {"script", "style", "nav", "footer", "header",
                 "aside", "iframe", "noscript", "svg"}

    async def scrape(self, url: str) -> Tuple[str, str, str]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgenticBot/1.0)"}
        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            canonical_url = str(response.url)
        soup = BeautifulSoup(response.text, "html.parser")
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)
        for tag in soup(self.SKIP_TAGS):
            tag.decompose()
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("body")
        )
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s{3,}", "  ", text)
        return title, text.strip(), canonical_url

    async def discover_links(self, base_url: str, depth: int = 1) -> List[str]:
        from urllib.parse import urljoin, urlparse
        visited = set()
        to_visit = {base_url}
        base_domain = urlparse(base_url).netloc
        for _ in range(depth):
            next_batch = set()
            for url in to_visit:
                if url in visited:
                    continue
                visited.add(url)
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        resp = await client.get(url, follow_redirects=True)
                        soup = BeautifulSoup(resp.text, "html.parser")
                        for a in soup.find_all("a", href=True):
                            link = urljoin(url, a["href"])
                            parsed = urlparse(link)
                            if (
                                parsed.netloc == base_domain
                                and not parsed.fragment
                                and link not in visited
                            ):
                                next_batch.add(link)
                except Exception:
                    pass
            to_visit = next_batch
        return list(visited)


class PDFParser:

    def parse(self, file_bytes: bytes, filename: str) -> Tuple[str, str]:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            title = filename
            if reader.metadata and reader.metadata.title:
                title = reader.metadata.title
            pages_text = []
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages_text.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception:
                    pass
            full_text = "\n\n".join(pages_text)
            return title, full_text.strip()
        except Exception as e:
            raise ValueError(f"Failed to parse PDF '{filename}': {e}")


class IngestionPipeline:

    def __init__(self):
        self.chunker = SemanticChunker()
        self.scraper = WebScraper()
        self.pdf_parser = PDFParser()
        self.vector_store = get_vector_store()

    async def ingest_url(
        self,
        bot_id: str,
        url: str,
        source_id: str,
    ) -> Dict[str, Any]:
        title, text, canonical_url = await self.scraper.scrape(url)
        if not text:
            raise ValueError(f"No content extracted from URL: {url}")
        chunks = self.chunker.split(text)
        metadatas = [
            {
                "source_type": "url",
                "source_name": title or url,
                "source_url": canonical_url,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(len(chunks))
        ]
        count = self.vector_store.add_documents(
            bot_id=bot_id,
            chunks=chunks,
            metadatas=metadatas,
            source_id=source_id,
        )
        return {
            "source_id": source_id,
            "source_type": "url",
            "source_name": title or url,
            "source_url": canonical_url,
            "chunk_count": count,
        }

    async def ingest_pdf(
        self,
        bot_id: str,
        file_bytes: bytes,
        filename: str,
        source_id: str,
        source_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        title, text = self.pdf_parser.parse(file_bytes, filename)
        if not text:
            raise ValueError(f"No text extracted from PDF: {filename}")
        chunks = self.chunker.split(text)
        metadatas = [
            {
                "source_type": "pdf",
                "source_name": title or filename,
                "source_url": source_url or "",
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(len(chunks))
        ]
        count = self.vector_store.add_documents(
            bot_id=bot_id,
            chunks=chunks,
            metadatas=metadatas,
            source_id=source_id,
        )
        return {
            "source_id": source_id,
            "source_type": "pdf",
            "source_name": title or filename,
            "source_url": source_url,
            "chunk_count": count,
        }

    async def ingest_text(
        self,
        bot_id: str,
        title: str,
        content: str,
        source_id: str,
        source_url: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chunks = self.chunker.split(content)
        metadatas = [
            {
                "source_type": "text",
                "source_name": title,
                "source_url": source_url or "",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                **(extra_metadata or {}),
            }
            for i in range(len(chunks))
        ]
        count = self.vector_store.add_documents(
            bot_id=bot_id,
            chunks=chunks,
            metadatas=metadatas,
            source_id=source_id,
        )
        return {
            "source_id": source_id,
            "source_type": "text",
            "source_name": title,
            "source_url": source_url,
            "chunk_count": count,
        }
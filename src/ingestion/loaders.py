"""
Document Loaders for SentinelRAG.
Supports PDF, DOCX, Markdown (.md), and Plain Text (.txt) with rich metadata extraction.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re
import logging

logger = logging.getLogger("SentinelRAG.Ingestion")

@dataclass
class SentinelDocument:
    """Standardized document representation across all file types."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> Optional[int]:
        return self.metadata.get("page", None)

    @property
    def section(self) -> Optional[str]:
        return self.metadata.get("section", None)


def load_markdown_or_txt(file_path: Path) -> List[SentinelDocument]:
    """Loads markdown or plain text files, extracting headers as potential sections."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    
    # Check if there are markdown sections (## Header)
    sections = re.split(r'\n(?=##?\s+)', text)
    docs = []
    
    if len(sections) > 1:
        for idx, sec in enumerate(sections):
            stripped = sec.strip()
            if not stripped:
                continue
            # Extract header title if present
            header_match = re.match(r'^##?\s+(.*)', stripped)
            section_title = header_match.group(1).strip() if header_match else f"Section {idx+1}"
            
            docs.append(SentinelDocument(
                page_content=stripped,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path.resolve()),
                    "file_type": file_path.suffix.lower(),
                    "section": section_title,
                    "section_index": idx + 1,
                    "total_sections": len(sections)
                }
            ))
    else:
        docs.append(SentinelDocument(
            page_content=text.strip(),
            metadata={
                "source": file_path.name,
                "file_path": str(file_path.resolve()),
                "file_type": file_path.suffix.lower(),
                "section": "Main",
                "page": 1
            }
        ))
    return docs


def load_pdf(file_path: Path) -> List[SentinelDocument]:
    """Loads PDF documents page by page using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf is not installed. Please install it via pip.")
        return []

    reader = PdfReader(str(file_path))
    docs = []
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            docs.append(SentinelDocument(
                page_content=page_text.strip(),
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path.resolve()),
                    "file_type": ".pdf",
                    "page": page_idx + 1,
                    "total_pages": len(reader.pages)
                }
            ))
    return docs


def load_docx(file_path: Path) -> List[SentinelDocument]:
    """Loads DOCX documents using python-docx."""
    try:
        import docx
    except ImportError:
        logger.error("python-docx is not installed. Please install it via pip.")
        return []

    doc = docx.Document(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)

    return [
        SentinelDocument(
            page_content=full_text,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path.resolve()),
                "file_type": ".docx",
                "section": "Main",
                "paragraph_count": len(paragraphs)
            }
        )
    ]


def load_document(file_path: Path) -> List[SentinelDocument]:
    """Dispatches document loading to the appropriate loader based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix in [".md", ".markdown", ".txt"]:
        return load_markdown_or_txt(file_path)
    elif suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix in [".docx", ".doc"]:
        return load_docx(file_path)
    else:
        logger.warning(f"Unsupported file format: {file_path}. Skipping.")
        return []


def load_documents_from_directory(directory_path: Path) -> List[SentinelDocument]:
    """Recursively or flatly loads all supported documents from a directory."""
    supported_extensions = {".md", ".markdown", ".txt", ".pdf", ".docx"}
    all_docs = []
    
    if not directory_path.exists():
        logger.warning(f"Directory {directory_path} does not exist.")
        return []

    for item in directory_path.iterdir():
        if item.is_file() and item.suffix.lower() in supported_extensions:
            docs = load_document(item)
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} sub-sections/pages from {item.name}")

    return all_docs

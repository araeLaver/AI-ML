# -*- coding: utf-8 -*-
"""
Data Management Module

- Sample Documents (DART 스타일)
- Document Loader (PDF/TXT 업로드)
"""

from .sample_docs import SAMPLE_DOCUMENTS, get_all_documents
from .document_loader import DocumentLoader, Document

__all__ = [
    "SAMPLE_DOCUMENTS",
    "get_all_documents",
    "DocumentLoader",
    "Document",
]

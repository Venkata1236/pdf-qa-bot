# core/loader.py
# Loads PDF files and extracts raw text
# Concept: Document Loaders

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
import tempfile
import os


def load_pdf_from_path(file_path: str) -> List[Document]:
    """
    Loads a PDF from a file path.
    Returns a list of Document objects (one per page).

    Args:
        file_path: path to the PDF file

    Returns:
        List of Document objects with page_content and metadata
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} pages from PDF")
    return documents


def load_pdf_from_bytes(file_bytes: bytes, filename: str = "upload.pdf") -> List[Document]:
    """
    Loads a PDF from raw bytes (for Streamlit file uploader).
    Saves to a temp file first, then loads.

    Args:
        file_bytes: raw bytes of the PDF
        filename: name of the file

    Returns:
        List of Document objects
    """
    # Save bytes to a temp file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        documents = load_pdf_from_path(tmp_path)
        return documents
    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)


def get_total_text(documents: List[Document]) -> str:
    """
    Combines all document pages into one string.
    Useful for quick preview.

    Args:
        documents: list of Document objects

    Returns:
        Full text as single string
    """
    return "\n\n".join([doc.page_content for doc in documents])
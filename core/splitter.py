# core/splitter.py
# Splits loaded documents into smaller chunks
# Concept: Text Splitters

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits documents into smaller chunks for embedding.

    Why we split:
    - OpenAI has a token limit per API call
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries

    Args:
        documents   : list of Document objects from loader
        chunk_size  : max characters per chunk (default 1000)
        chunk_overlap: characters shared between chunks (default 200)

    Returns:
        List of smaller Document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
        # Tries to split on paragraphs first,
        # then lines, then sentences, then words
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Split {len(documents)} pages into {len(chunks)} chunks")
    print(f"   chunk_size={chunk_size} | chunk_overlap={chunk_overlap}")

    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Returns stats about the chunks for debugging.

    Args:
        chunks: list of Document chunks

    Returns:
        dict with min, max, avg chunk sizes
    """
    sizes = [len(chunk.page_content) for chunk in chunks]

    return {
        "total_chunks": len(chunks),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": int(sum(sizes) / len(sizes))
    }
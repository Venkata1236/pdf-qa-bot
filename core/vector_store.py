# core/vector_store.py
# Stores and searches embeddings using FAISS
# Concept: Vector Store — semantic search engine

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List


def create_vector_store(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings
) -> FAISS:
    """
    Creates a FAISS vector store from document chunks.

    What happens here:
    1. Each chunk is converted to a vector (1536 numbers)
    2. All vectors are stored in FAISS index
    3. FAISS enables fast similarity search

    Args:
        chunks    : list of Document chunks from splitter
        embeddings: OpenAIEmbeddings model

    Returns:
        FAISS vector store ready for searching
    """
    print(f"⏳ Creating vector store from {len(chunks)} chunks...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print(f"✅ Vector store created successfully!")
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = "faiss_index"):
    """
    Saves the FAISS index to disk.
    Useful so you don't re-embed the same PDF every time.

    Args:
        vector_store: FAISS vector store to save
        path        : folder path to save the index
    """
    vector_store.save_local(path)
    print(f"✅ Vector store saved to '{path}/'")


def load_vector_store(
    path: str,
    embeddings: OpenAIEmbeddings
) -> FAISS:
    """
    Loads a previously saved FAISS index from disk.

    Args:
        path      : folder path where index was saved
        embeddings: same embeddings model used to create it

    Returns:
        Loaded FAISS vector store
    """
    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Vector store loaded from '{path}/'")
    return vector_store


def search_similar_chunks(
    vector_store: FAISS,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Searches for the k most relevant chunks for a query.

    Args:
        vector_store: FAISS vector store
        query       : user's question
        k           : number of chunks to return (default 4)

    Returns:
        List of most relevant Document chunks
    """
    results = vector_store.similarity_search(query, k=k)
    print(f"✅ Found {len(results)} relevant chunks for query")
    return results

def merge_vector_stores(
    vector_stores: list,
    embeddings: OpenAIEmbeddings
) -> FAISS:
    """
    Merges multiple FAISS vector stores into one.
    Used for multi-PDF support.

    Args:
        vector_stores: list of FAISS vector stores
        embeddings   : same embeddings model used to create them

    Returns:
        Single merged FAISS vector store
    """
    merged = vector_stores[0]
    for vs in vector_stores[1:]:
        merged.merge_from(vs)

    print(f"✅ Merged {len(vector_stores)} vector stores into one")
    return merged